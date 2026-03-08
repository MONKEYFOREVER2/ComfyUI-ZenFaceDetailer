import torch
import numpy as np
import cv2
import os
import comfy.model_management as model_management
import comfy.samplers
import comfy.sd
import comfy.utils
import folder_paths
from nodes import common_ksampler

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
                "blur_mask": ("INT", {"default": 16, "min": 0, "max": 100, "step": 1}),
                
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
                bbox_threshold, bbox_dilation, bbox_crop_factor, drop_size, blur_mask,
                noise_type_init, eta, sampler_mode, sampler_name, scheduler, steps, cfg, denoise, seed):
                
        print("🎭 ZenFaceDetailer: Starting processing...")
        batch_size, height, width, channels = image.shape
        result_images = []
        result_masks = []
        
        for b in range(batch_size):
            img_tensor = image[b] # (H, W, C)
            img_tensor_batch = img_tensor.unsqueeze(0).movedim(-1, 1) # Comfy standard B, C, H, W for detectors usually, but impact wants B, H, W, C
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
                
                # Expand slightly over the crop region because Impact Pack crop_region is exact bbox with crop_factor applied
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
                
                # Local Feathered Mask
                local_mask = np.ones((crop_h, crop_w), dtype=np.float32)
                if blur_mask > 0:
                    local_mask_8u = (local_mask * 255).astype(np.uint8)
                    cv2.rectangle(local_mask_8u, (0,0), (crop_w-1, crop_h-1), 0, blur_mask * 2)
                    local_mask = cv2.GaussianBlur(local_mask_8u, (blur_mask*4|1, blur_mask*4|1), 0).astype(np.float32) / 255.0
                
                local_mask_3c = np.repeat(local_mask[:, :, np.newaxis], 3, axis=2)
                
                # Composite
                original_crop = output_img_np[y1:y2, x1:x2]
                blended = (decoded_np * local_mask_3c + original_crop * (1 - local_mask_3c)).astype(np.uint8)
                
                output_img_np[y1:y2, x1:x2] = blended
                full_mask[y1:y2, x1:x2] = np.maximum(full_mask[y1:y2, x1:x2], local_mask)
                
            out_tensor = torch.from_numpy(output_img_np.astype(np.float32) / 255.0)
            result_images.append(out_tensor)
            result_masks.append(torch.from_numpy(full_mask))
            
        return (torch.stack(result_images, dim=0), torch.stack(result_masks, dim=0))
