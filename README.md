# ComfyUI-MLX-DLSS

ComfyUI custom nodes for NVIDIA DLSS 5 Neural Rendering, Frame Generation, and RTX Video Super Resolution (VSR), ported from [iamwavecut/MLX-DLSS](https://github.com/iamwavecut/MLX-DLSS).

Runs directly in PyTorch with CUDA acceleration (also supports CPU and Apple Silicon MPS).

## Features
- **DLSS5 Neural Rendering (Image)**:
  - Enhance image details, tone, and sharpness using the recovered DLSS 5 neural rendering transformer.
  - Controls matching `mlxdlss-web`: `profile` (standard, natural, cinematic, neutral), `intensity`, `detail_strength`, `colour_strength`, `detail_radius`, `processing_scale`.
  - Toggle 2× upscaling via RTX Video Super Resolution (`enable_vsr`).
- **DLSS5 Neural Rendering (Video)**:
  - Temporal stability with dense optical flow and scene-cut detection.
  - DLSS Frame Generation (`enable_framegen`) with factors ×2, ×3, ×4, ×8, ×16 and configurable order (`nr_first` or `fg_first`).
  - Toggle 2× upscaling via RTX Video Super Resolution (`enable_vsr`).
  - Native ComfyUI progress bar updates and interruption support.

## Models & Weights
Place the following `.safetensors` files in `ComfyUI/models/dlss/`:
1. `dlssnr-weights-logical.safetensors`: Recovered logical weights for DLSS 5 Neural Rendering.
2. `framegen.safetensors`: DLSS Frame Generation weights.
3. `vsr.safetensors`: RTX Video Super Resolution 1.8.2 weights.
