import torch
import numpy as np
import cv2
import comfy.model_management as model_management
import comfy.samplers
import comfy.sd
import comfy.utils
import folder_paths
from nodes import common_ksampler


def _create_elliptical_mask(h, w):
    """Create an elliptical binary mask (1 inside, 0 outside)."""
    mask = np.zeros((h, w), dtype=np.float32)
    cx, cy = w // 2, h // 2
    cv2.ellipse(mask, (cx, cy), (cx, cy), 0, 0, 360, 1.0, -1)
    return mask


def _create_squircle_mask(h, w, n=4):
    """Create a superellipse (squircle) mask with exponent n."""
    y = np.linspace(-1, 1, h).reshape(-1, 1)
    x = np.linspace(-1, 1, w).reshape(1, -1)
    d = np.abs(x) ** n + np.abs(y) ** n
    mask = np.where(d <= 1.0, 1.0, 0.0).astype(np.float32)
    return mask


def _create_rectangle_mask(h, w):
    """Create a full rectangle mask."""
    return np.ones((h, w), dtype=np.float32)


def _feather_mask(mask, feather_px):
    """Apply gaussian feathering to the edges of a binary mask."""
    if feather_px <= 0:
        return mask
    # Use distance transform for smooth falloff from edges
    # Invert mask to get distance from edge (outside->inside)
    mask_u8 = (mask * 255).astype(np.uint8)
    dist = cv2.distanceTransform(mask_u8, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    # Normalize so that pixels at feather_px distance are fully opaque
    feathered = np.clip(dist / max(feather_px, 1), 0.0, 1.0).astype(np.float32)
    return feathered


def _expand_mask(mask, pixels):
    """Dilate (positive) or erode (negative) a binary mask."""
    if pixels == 0:
        return mask
    kernel_size = abs(pixels) * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask_u8 = (mask * 255).astype(np.uint8)
    if pixels > 0:
        result = cv2.dilate(mask_u8, kernel, iterations=1)
    else:
        result = cv2.erode(mask_u8, kernel, iterations=1)
    return result.astype(np.float32) / 255.0


def _match_color(source, target):
    """Match the color statistics of source to target (per-channel mean/std transfer)."""
    result = source.copy().astype(np.float32)
    for c in range(3):
        s_mean, s_std = result[:, :, c].mean(), result[:, :, c].std() + 1e-6
        t_mean, t_std = target[:, :, c].mean(), target[:, :, c].std() + 1e-6
        result[:, :, c] = (result[:, :, c] - s_mean) * (t_std / s_std) + t_mean
    return np.clip(result, 0, 255).astype(np.uint8)


def _blend_soft_light(base, detail, alpha):
    """Soft light blend mode with alpha mask."""
    base_f = base.astype(np.float32) / 255.0
    detail_f = detail.astype(np.float32) / 255.0
    # Pegtop soft light formula
    blended = (1 - 2 * detail_f) * base_f * base_f + 2 * detail_f * base_f
    blended = np.clip(blended, 0, 1)
    # Mix using alpha
    result = base_f * (1 - alpha) + blended * alpha
    return (np.clip(result, 0, 1) * 255).astype(np.uint8)


def _blend_overlay(base, detail, alpha):
    """Overlay blend mode with alpha mask."""
    base_f = base.astype(np.float32) / 255.0
    detail_f = detail.astype(np.float32) / 255.0
    # Overlay formula
    low = 2 * base_f * detail_f
    high = 1 - 2 * (1 - base_f) * (1 - detail_f)
    blended = np.where(base_f < 0.5, low, high)
    blended = np.clip(blended, 0, 1)
    result = base_f * (1 - alpha) + blended * alpha
    return (np.clip(result, 0, 1) * 255).astype(np.uint8)


def _blend_normal(base, detail, alpha):
    """Normal alpha blend."""
    return (detail * alpha + base * (1 - alpha)).astype(np.uint8)


class ZenFaceDetailer:
    @classmethod
    def INPUT_TYPES(s):
        samplers = comfy.samplers.KSampler.SAMPLERS
        schedulers = comfy.samplers.KSampler.SCHEDULERS
        
        return {
            "required": {
                "image": ("IMAGE",),
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "bbox_detector": ("BBOX_DETECTOR", ),
                
                # FaceDetailer Detector Settings
                "bbox_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "bbox_dilation": ("INT", {"default": 10, "min": -512, "max": 512, "step": 1}),
                "bbox_crop_factor": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 10.0, "step": 0.1}),
                "drop_size": ("INT", {"default": 10, "min": 1, "max": 10000, "step": 1}),
                
                # Mask & Feathering
                "mask_shape": (["ellipse", "squircle", "rectangle"],),
                "feather_amount": ("INT", {"default": 40, "min": 0, "max": 200, "step": 1}),
                "mask_expand": ("INT", {"default": 0, "min": -100, "max": 100, "step": 1}),
                
                # Blending
                "blend_mode": (["normal", "soft_light", "overlay"],),
                "color_match": ("BOOLEAN", {"default": True}),
                
                # ClownsharKSampler-like Noise / Advanced Sampler Settings
                "noise_type_init": (["default", "gaussian", "uniform", "perlin", "simplex"],),
                "eta": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "sampler_mode": (["standard", "unsample", "resample"],),
                "sampler_name": (samplers, ),
                "scheduler": (schedulers, ),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "denoise": ("FLOAT", {"default": 0.45, "min": 0.0, "max": 1.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }
        
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "process"
    CATEGORY = "Zen/Detailers"

    def process(self, image, model, clip, vae, positive, negative, bbox_detector,
                bbox_threshold, bbox_dilation, bbox_crop_factor, drop_size,
                mask_shape, feather_amount, mask_expand,
                blend_mode, color_match,
                noise_type_init, eta, sampler_mode, sampler_name, scheduler, steps, cfg, denoise, seed):
                
        print("🎭 ZenFaceDetailer: Starting processing...")
        batch_size, height, width, channels = image.shape
        result_images = []
        result_masks = []
        
        for b in range(batch_size):
            img_tensor = image[b] # (H, W, C)
            # Impact pack detect takes B, H, W, C
            img_input = img_tensor.unsqueeze(0)
            
            # Detect faces using BBOX_DETECTOR
            if hasattr(bbox_detector, 'detect'):
                try:
                    segs = bbox_detector.detect(img_input, bbox_threshold, bbox_dilation, bbox_crop_factor, drop_size)
                except TypeError:
                    # Fallback for signatures that require detailer_hook
                    segs = bbox_detector.detect(img_input, bbox_threshold, bbox_dilation, bbox_crop_factor, drop_size, None)
            else:
                print("🎭 ZenFaceDetailer: Invalid BBOX_DETECTOR object.")
                result_images.append(img_tensor)
                result_masks.append(torch.zeros((height, width), dtype=torch.float32, device="cpu"))
                continue
                
            seg_list = segs[1] if isinstance(segs, tuple) and len(segs) > 1 else []
            
            if not seg_list:
                print(f"🎭 ZenFaceDetailer: No faces detected in batch index {b}.")
                result_images.append(img_tensor)
                result_masks.append(torch.zeros((height, width), dtype=torch.float32, device="cpu"))
                continue
                
            print(f"🎭 ZenFaceDetailer: Detected {len(seg_list)} faces.")
            img_np = (img_tensor.cpu().numpy() * 255.0).astype(np.uint8)
            output_img_np = img_np.copy()
            full_mask = np.zeros((height, width), dtype=np.float32)
            
            for seg in seg_list:
                crop_region = getattr(seg, 'crop_region', None)
                if crop_region is None:
                    continue
                    
                x1, y1, x2, y2 = map(int, crop_region)
                
                # Make sure crop dimensions are multiples of 8 for VAE
                crop_w = x2 - x1
                crop_h = y2 - y1
                
                pad_w = (8 - crop_w % 8) % 8
                pad_h = (8 - crop_h % 8) % 8
                
                if x2 + pad_w <= width: x2 += pad_w
                elif x1 - pad_w >= 0: x1 -= pad_w
                
                if y2 + pad_h <= height: y2 += pad_h
                elif y1 - pad_h >= 0: y1 -= pad_h
                
                crop_w = x2 - x1
                crop_h = y2 - y1
                
                # Extract Crop
                crop_np = output_img_np[y1:y2, x1:x2]
                crop_tensor = torch.from_numpy(crop_np.astype(np.float32) / 255.0).unsqueeze(0)
                
                print(f"🎭 ZenFaceDetailer: VAE Encoding face region {crop_w}x{crop_h}")
                encoded = vae.encode(crop_tensor[:, :, :, :3])
                
                disable_noise = sampler_mode in ["unsample", "resample"]
                    
                print(f"🎭 ZenFaceDetailer: Sampling with {sampler_name} (Mode: {sampler_mode}, ETA: {eta})")
                
                try:
                    latent_input = {"samples": encoded}
                    sampled = common_ksampler(
                        model, seed, steps, cfg, sampler_name, scheduler, 
                        positive, negative, latent_input, disable_noise=disable_noise, 
                        start_step=None, last_step=None, force_full_denoise=False, denoise=denoise
                    )
                    sampled_tensor = sampled[0]["samples"]
                except Exception as e:
                    import traceback
                    print(f"🎭 ZenFaceDetailer: Sampling error - {e}")
                    traceback.print_exc()
                    sampled_tensor = encoded
                
                decoded = vae.decode(sampled_tensor)
                decoded_np = (decoded[0].cpu().numpy() * 255.0).astype(np.uint8)
                decoded_np = cv2.resize(decoded_np, (crop_w, crop_h))
                
                # --- Color matching ---
                if color_match:
                    decoded_np = _match_color(decoded_np, crop_np)
                
                # --- Shaped mask generation ---
                if mask_shape == "ellipse":
                    local_mask = _create_elliptical_mask(crop_h, crop_w)
                elif mask_shape == "squircle":
                    local_mask = _create_squircle_mask(crop_h, crop_w, n=4)
                else:
                    local_mask = _create_rectangle_mask(crop_h, crop_w)
                
                # --- Expand / shrink ---
                if mask_expand != 0:
                    local_mask = _expand_mask(local_mask, mask_expand)
                
                # --- Feathering ---
                local_mask = _feather_mask(local_mask, feather_amount)
                
                # --- Blending ---
                local_mask_3c = np.repeat(local_mask[:, :, np.newaxis], 3, axis=2)
                original_crop = output_img_np[y1:y2, x1:x2]
                
                if blend_mode == "soft_light":
                    blended = _blend_soft_light(original_crop, decoded_np, local_mask_3c)
                elif blend_mode == "overlay":
                    blended = _blend_overlay(original_crop, decoded_np, local_mask_3c)
                else:
                    blended = _blend_normal(original_crop, decoded_np, local_mask_3c)
                
                output_img_np[y1:y2, x1:x2] = blended
                full_mask[y1:y2, x1:x2] = np.maximum(full_mask[y1:y2, x1:x2], local_mask)
                
            out_tensor = torch.from_numpy(output_img_np.astype(np.float32) / 255.0)
            result_images.append(out_tensor)
            result_masks.append(torch.from_numpy(full_mask))
            
        return (torch.stack(result_images, dim=0), torch.stack(result_masks, dim=0))
