"""Model manager and caching for DLSS pipelines in ComfyUI."""
from __future__ import annotations

import os
import pathlib
from typing import Any, Tuple
import torch

try:
    import folder_paths
except ImportError:
    folder_paths = None

from .pipeline import NeuralRenderingPipeline
from .framegen import FrameGenerator
from .vsr import VideoSuperResolver

_PIPELINE_CACHE: dict[Tuple[str, str, str], NeuralRenderingPipeline] = {}
_FRAMEGEN_CACHE: dict[Tuple[str, str, str], FrameGenerator] = {}
_VSR_CACHE: dict[Tuple[str, str, str], VideoSuperResolver] = {}


def resolve_model_path(filename: str, folder_name: str = "dlss") -> str:
    """Find the full absolute path for a model file."""
    if not filename:
        return ""

    # 1. If it's already an existing path
    p = pathlib.Path(filename).expanduser()
    if p.is_file():
        return str(p.resolve())

    # 2. Check ComfyUI folder_paths if available
    if folder_paths is not None:
        try:
            full_path = folder_paths.get_full_path(folder_name, filename)
            if full_path and os.path.isfile(full_path):
                return str(pathlib.Path(full_path).resolve())
        except Exception:
            pass

        try:
            full_path = folder_paths.get_full_path("upscale_models", filename)
            if full_path and os.path.isfile(full_path):
                return str(pathlib.Path(full_path).resolve())
        except Exception:
            pass

        # Models base dir
        models_dir = getattr(folder_paths, "models_dir", None)
        if models_dir:
            candidate = os.path.join(models_dir, folder_name, filename)
            if os.path.isfile(candidate):
                return str(pathlib.Path(candidate).resolve())

    # 3. Check common local paths
    home_dir = str(pathlib.Path.home())
    fallback_paths = [
        os.path.join(home_dir, filename),
        os.path.join(home_dir, "ComfyUI", "models", "dlss", filename),
        os.path.join(os.path.dirname(__file__), "..", "..", "models", "dlss", filename),
    ]
    for fb in fallback_paths:
        if os.path.isfile(fb):
            return str(pathlib.Path(fb).resolve())

    raise FileNotFoundError(
        f"Model weight file '{filename}' was not found. "
        f"Please ensure it is placed in 'models/dlss/'."
    )


def get_neural_rendering_pipeline(
    weights_filename: str,
    device: str | torch.device = "auto",
    precision: str = "fast",
) -> NeuralRenderingPipeline:
    path = resolve_model_path(weights_filename, "dlss")
    dev_str = str(device)
    cache_key = (path, dev_str, precision)
    if cache_key not in _PIPELINE_CACHE:
        _PIPELINE_CACHE[cache_key] = NeuralRenderingPipeline.from_safetensors(
            path, device=device, precision=precision
        )
    return _PIPELINE_CACHE[cache_key]


def get_frame_generator(
    weights_filename: str,
    device: str | torch.device = "auto",
    precision: str = "fast",
    use_tensorrt: bool = False,
) -> FrameGenerator:
    path = resolve_model_path(weights_filename, "dlss")
    dev_str = str(device)
    cache_key = (path, dev_str, precision, use_tensorrt)
    if cache_key not in _FRAMEGEN_CACHE:
        _FRAMEGEN_CACHE[cache_key] = FrameGenerator.from_safetensors(
            path, device=device, precision=precision, use_tensorrt=use_tensorrt
        )
    return _FRAMEGEN_CACHE[cache_key]


def get_vsr_resolver(
    weights_filename: str,
    device: str | torch.device = "auto",
    precision: str = "fast",
    use_tensorrt: bool = False,
) -> VideoSuperResolver:
    path = resolve_model_path(weights_filename, "dlss")
    dev_str = str(device)
    cache_key = (path, dev_str, precision, use_tensorrt)
    if cache_key not in _VSR_CACHE:
        _VSR_CACHE[cache_key] = VideoSuperResolver.from_safetensors(
            path, device=device, precision=precision, use_tensorrt=use_tensorrt
        )
    return _VSR_CACHE[cache_key]
