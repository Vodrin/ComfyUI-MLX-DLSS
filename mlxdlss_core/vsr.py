"""PyTorch implementation of NVIDIA RTX Video Super Resolution (VSR 1.8.2 High Bitrate Low 2x).

Upscales RGB in [0, 1] by 2x using the weights from vsr.safetensors.
Supports both image frames and batched video tensors on CUDA, CPU, and MPS.
"""
from __future__ import annotations

import pathlib
from typing import Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file

from .pipeline import resolve_device


class VSRBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv0 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c0 = F.elu(self.conv0(x), alpha=1.0)
        c1 = F.elu(self.conv1(c0), alpha=1.0)
        sc = self.shortcut(x)
        return c1 + sc


class VSRModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoders
        self.encoder0 = VSRBlock(16, 16)
        self.encoder1 = VSRBlock(16, 32)
        self.encoder2 = VSRBlock(32, 32)
        self.encoder3 = VSRBlock(32, 64)
        self.encoder4 = VSRBlock(64, 64)

        # Decoders
        self.decoder0_upsample = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.decoder0 = VSRBlock(128, 64)

        self.decoder1_upsample = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.decoder1 = VSRBlock(64, 32)

        self.decoder2_upsample = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.decoder2 = VSRBlock(64, 32)

        self.decoder3_upsample = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.decoder3 = VSRBlock(32, 16)

        # Output layers
        self.output_conv = nn.Conv2d(16, 64, kernel_size=3, padding=1)
        self.output_project = nn.Conv2d(64, 48, kernel_size=1, padding=0)

    def load_safetensors_weights(self, weights: dict[str, torch.Tensor]):
        state = {}
        for k, v in weights.items():
            if "upsample" in k:
                mapped = k.replace(".upsample.", "_upsample.")
            elif "output." in k:
                mapped = k.replace("output.", "output_")
            else:
                mapped = k
            state[mapped] = v
        self.load_state_dict(state, strict=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        # Encoder 0..3 with 2x average pooling
        x = self.encoder0(x); skips.append(x); x = F.avg_pool2d(x, 2)
        x = self.encoder1(x); skips.append(x); x = F.avg_pool2d(x, 2)
        x = self.encoder2(x); skips.append(x); x = F.avg_pool2d(x, 2)
        x = self.encoder3(x); skips.append(x); x = F.avg_pool2d(x, 2)

        # Bottleneck
        x = self.encoder4(x)

        # Decoders with nearest neighbor 2x upsample + conv + ELU + skip concat
        up0 = F.elu(self.decoder0_upsample(F.interpolate(x, scale_factor=2, mode="nearest")), 1.0)
        x = self.decoder0(torch.cat([up0, skips.pop()], dim=1))

        up1 = F.elu(self.decoder1_upsample(F.interpolate(x, scale_factor=2, mode="nearest")), 1.0)
        x = self.decoder1(torch.cat([up1, skips.pop()], dim=1))

        up2 = F.elu(self.decoder2_upsample(F.interpolate(x, scale_factor=2, mode="nearest")), 1.0)
        x = self.decoder2(torch.cat([up2, skips.pop()], dim=1))

        up3 = F.elu(self.decoder3_upsample(F.interpolate(x, scale_factor=2, mode="nearest")), 1.0)
        x = self.decoder3(torch.cat([up3, skips.pop()], dim=1))

        # Output head
        x = F.elu(self.output_conv(x), 1.0)
        x = self.output_project(x)
        return torch.tanh(x)


class VideoSuperResolver:
    """High-level runner for RTX Video Super Resolution 2x upscaling."""

    def __init__(
        self,
        weights: dict[str, torch.Tensor],
        *,
        device: str | torch.device = "auto",
        precision: str = "fast",
        use_tensorrt: bool = False,
        weights_path: str | pathlib.Path | None = None,
    ):
        self.device = resolve_device(device)
        self.precision = precision
        self.use_tensorrt = use_tensorrt
        self.trt_runner = None
        self.model = VSRModel()
        self.model.load_safetensors_weights(weights)
        self.model.eval()

        self.dtype = torch.float32
        if precision == "fast" and self.device.type != "cpu":
            self.dtype = torch.float16
            self.model = self.model.to(torch.float16)

        self.model = self.model.to(self.device)

        if use_tensorrt and self.device.type == "cuda" and weights_path:
            try:
                from .tensorrt_backend import is_tensorrt_available, get_or_build_vsr_trt_runner
                if is_tensorrt_available():
                    self.trt_runner = get_or_build_vsr_trt_runner(weights_path, self.device)
            except Exception as e:
                import warnings
                warnings.warn(f"TensorRT initialization failed, falling back to PyTorch CUDA: {e}")

    @classmethod
    def from_safetensors(
        cls,
        path: str | pathlib.Path,
        use_tensorrt: bool = False,
        **kwargs: Any,
    ) -> "VideoSuperResolver":
        return cls(load_file(str(path)), weights_path=path, use_tensorrt=use_tensorrt, **kwargs)

    @torch.no_grad()
    def upscale(self, rgb: torch.Tensor | np.ndarray) -> torch.Tensor:
        """Upscales RGB image [N, C, H, W] or [H, W, 3] in [0, 1] by 2x.

        Returns [N, C, 2H, 2W] tensor on self.device clamped to [0, 1].
        """
        if isinstance(rgb, np.ndarray):
            tensor = torch.from_numpy(np.ascontiguousarray(rgb))
            if tensor.ndim == 3 and tensor.shape[-1] == 3:
                tensor = tensor.permute(2, 0, 1)[None]
            elif tensor.ndim == 4 and tensor.shape[-1] == 3:
                tensor = tensor.permute(0, 3, 1, 2)
        else:
            tensor = rgb
            if tensor.ndim == 3 and tensor.shape[-1] == 3:
                tensor = tensor.permute(2, 0, 1)[None]
            elif tensor.ndim == 4 and tensor.shape[-1] == 3:
                tensor = tensor.permute(0, 3, 1, 2)

        N, C, orig_H, orig_W = tensor.shape
        if C != 3:
            raise ValueError(f"Expected 3-channel RGB, got {C} channels")

        H = (orig_H + 31) // 32 * 32
        W = (orig_W + 31) // 32 * 32

        tensor = tensor.to(self.device, torch.float32)
        quantized = torch.round(torch.clamp(tensor, 0.0, 1.0) * 255.0) / 255.0
        base = (quantized - 0.5) * 2.0

        if H > orig_H or W > orig_W:
            base = F.pad(base, (0, W - orig_W, 0, H - orig_H), mode="replicate")

        # RGBA padding with zero alpha: [N, 4, H, W]
        rgba = F.pad(base, (0, 0, 0, 0, 0, 1))

        # Space-to-depth 2x: (N, 4, H, W) -> (N, 16, H//2, W//2)
        rgba_grid = rgba.view(N, 4, H // 2, 2, W // 2, 2)
        x = rgba_grid.permute(0, 3, 5, 1, 2, 4).reshape(N, 16, H // 2, W // 2)

        # Run network
        x = x.to(self.dtype)
        if self.trt_runner is not None:
            out = self.trt_runner(x).to(torch.float32)
        else:
            out = self.model(x).to(torch.float32)  # [N, 48, H//2, W//2]

        # Depth-to-space: (N, 48, H//2, W//2) -> (N, 3, H*2, W*2)
        res_grid = out.view(N, 3, 4, 4, H // 2, W // 2)
        residual = res_grid.permute(0, 1, 4, 2, 5, 3).reshape(N, 3, H * 2, W * 2)

        # Base bicubic upsample 2x
        base_up = F.interpolate(base, scale_factor=2, mode="bicubic", align_corners=False)
        output = torch.clamp((base_up + residual) * 0.5 + 0.5, 0.0, 1.0)
        output = torch.floor(output * 255.0) / 255.0

        # Crop back to original dimensions * 2
        return output[:, :, : orig_H * 2, : orig_W * 2]
