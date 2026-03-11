# 🎭 ZenFace Detailer (Clownshark Edition)

An all-in-one ComfyUI face detailing node that combines **face detection** + **inpainting** + **advanced sampling** in a single node — no complex node chains required.

> Detects faces using Impact Pack's BBOX detector, crops each face, re-samples with your chosen sampler/scheduler, and composites back with feathered blending. One node does it all.

---

## ✨ Features

- **Automatic face detection** via Impact Pack's BBOX_DETECTOR
- **Per-face VAE encode → KSample → VAE decode** pipeline
- **Advanced sampler controls** — choose any sampler, scheduler, noise type, and sampling mode (standard/unsample/resample)
- **Shaped mask blending** — ellipse, squircle, or rectangle masks for natural face-shaped transitions
- **Smooth feathering** — distance-transform-based feathering for artifact-free compositing, even at high denoise
- **Blend modes** — normal, soft light, and overlay blending for seamless integration
- **Color matching** — automatic per-channel color statistics transfer prevents color shifts at high denoise
- **Mask expand/shrink** — fine-tune the mask boundary before feathering
- **Batch processing** — handles multiple images in a batch
- **Outputs both image and mask** — use the mask for further processing downstream
- **VAE-aligned crops** — automatically pads face crops to multiples of 8

---

## 🚀 Installation

### Prerequisites

This node requires [ComfyUI Impact Pack](https://github.com/ltdrdata/ComfyUI-Impact-Pack) to be installed (provides the BBOX_DETECTOR).

### Option 1: Git Clone (Recommended)

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/MONKEYFOREVER2/ComfyUI-ZenFaceDetailer.git
cd ComfyUI-ZenFaceDetailer
pip install -r requirements.txt
```

Restart ComfyUI.

### Option 2: Manual Download

1. Click the green **Code** button → **Download ZIP**
2. Extract into `ComfyUI/custom_nodes/`
3. Rename folder to `ComfyUI-ZenFaceDetailer` if needed
4. Install dependencies:
   ```bash
   cd ComfyUI/custom_nodes/ComfyUI-ZenFaceDetailer
   pip install -r requirements.txt
   ```
5. Restart ComfyUI

### Option 3: ComfyUI Manager

1. Open ComfyUI Manager → **Install Custom Nodes**
2. Search for `ZenFace Detailer`
3. Click **Install** → Restart ComfyUI

---

## 🎯 Finding the Node

**Right-click** → **Add Node** → **Zen/Detailers** → **🎭 ZenFace Detailer (Clownshark Edition)**

---

## ⚡ Quick Start

1. Connect your **MODEL**, **CLIP**, **VAE**, **positive/negative conditioning**, and **image**
2. Connect a **BBOX_DETECTOR** (e.g. from Impact Pack's `UltralyticsDetectorProvider`)
3. That's it — defaults are tuned for good face detailing out of the box

### Recommended Workflow

```
[Generate Image] → [🎭 ZenFace Detailer] → [SaveImage]
                          ↑
              [UltralyticsDetectorProvider] (bbox_detector)
```

---

## 🔧 Parameters

### Inputs (Required Connections)
| Input | Type | Description |
|-------|------|-------------|
| `image` | IMAGE | The image to process |
| `model` | MODEL | Your checkpoint model |
| `clip` | CLIP | CLIP model for conditioning |
| `vae` | VAE | VAE for encode/decode |
| `positive` | CONDITIONING | Positive prompt conditioning |
| `negative` | CONDITIONING | Negative prompt conditioning |
| `bbox_detector` | BBOX_DETECTOR | Face detector from Impact Pack |

### Face Detection Settings
| Parameter | Default | Description |
|-----------|---------|-------------|
| `bbox_threshold` | `0.5` | Detection confidence threshold (0–1) |
| `bbox_dilation` | `10` | Expand/shrink detected bbox (pixels) |
| `bbox_crop_factor` | `3.0` | How much area around the face to crop |
| `drop_size` | `10` | Minimum face size to process (pixels) |

### Mask & Feathering Settings
| Parameter | Default | Description |
|-----------|---------|-------------|
| `mask_shape` | `ellipse` | Shape of the compositing mask: **ellipse** (natural face shape), **squircle** (rounded rectangle), or **rectangle** |
| `feather_amount` | `40` | Pixel radius of the soft feathered falloff (0–200). Higher = softer edges |
| `mask_expand` | `0` | Expand (positive) or shrink (negative) the mask before feathering (-100 to 100) |

### Blending Settings
| Parameter | Default | Description |
|-----------|---------|-------------|
| `blend_mode` | `normal` | How the detailed face blends: **normal** (standard alpha), **soft_light** (subtle, preserves shadows), **overlay** (contrasty, punchy) |
| `color_match` | `True` | Match color statistics of the detailed region to the original — prevents color shifts at high denoise |

### Sampler Settings
| Parameter | Default | Description |
|-----------|---------|-------------|
| `noise_type_init` | `default` | Noise type: default, gaussian, uniform, perlin, simplex |
| `eta` | `0.0` | Ancestral sampling noise factor |
| `sampler_mode` | `standard` | Sampling mode: standard, unsample, resample |
| `sampler_name` | — | KSampler sampler (euler, dpmpp_2m, etc.) |
| `scheduler` | — | KSampler scheduler (normal, karras, etc.) |
| `steps` | `20` | Number of sampling steps |
| `cfg` | `8.0` | Classifier-free guidance scale |
| `denoise` | `0.45` | Denoise strength (lower = preserve more detail) |
| `seed` | `0` | Random seed for reproducibility |

### Outputs
| Output | Type | Description |
|--------|------|-------------|
| `image` | IMAGE | Processed image with detailed faces |
| `mask` | MASK | Combined mask of all detected face regions |

---

## 💡 Blending Tips

| Scenario | Recommended Settings |
|----------|---------------------|
| **Low denoise (< 0.5)** | `normal` blend, `feather_amount` 20–40, `color_match` off |
| **High denoise (0.5–0.8)** | `soft_light` blend, `feather_amount` 60–100, `color_match` on |
| **Very high denoise (> 0.8)** | `soft_light` or `overlay`, `feather_amount` 100+, `mask_shape` ellipse, `color_match` on |
| **Multiple faces, different sizes** | Use `squircle` shape if ellipse clips edges on small faces |

---

## 📋 Requirements

- **ComfyUI** (latest version recommended)
- **[ComfyUI Impact Pack](https://github.com/ltdrdata/ComfyUI-Impact-Pack)** — provides face detection
- **opencv-python** — listed in `requirements.txt`
- **Python 3.10+**

---

## ⚠️ Upgrading from v1.0

v1.1 replaces `blur_mask` with the new `mask_shape`, `feather_amount`, `mask_expand`, `blend_mode`, and `color_match` parameters. If you have existing workflows using the old node, you'll need to delete and re-add the ZenFaceDetailer node, or manually update your workflow JSON.

---

## 🤝 Contributing

Pull requests welcome! Feel free to open issues or PRs.

## 📄 License

MIT License — free to use, modify, and distribute.
