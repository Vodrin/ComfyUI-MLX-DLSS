"""TensorRT backend and engine cache for DLSS models.

Provides on-demand ONNX export and TensorRT engine compilation, multi-profile
optimization (1MP, 1.5MP, 2MP), zero-copy PyTorch GPU inference via execute_async_v3,
and two-level (disk + VRAM) caching.
"""
from __future__ import annotations

import gc
import os
import pathlib
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# Global VRAM context cache: cache_key -> TensorRTEngineRunner
_TRT_RUNNER_CACHE: Dict[str, "TensorRTEngineRunner"] = {}


def is_tensorrt_available() -> bool:
    """Check if TensorRT and CUDA are both operational."""
    if not torch.cuda.is_available():
        return False
    try:
        import tensorrt as trt
        return True
    except ImportError:
        return False


def get_device_tag(device: Optional[torch.device] = None) -> str:
    """Generate a stable hardware identifier string (e.g. RTX_4090_sm89)."""
    if not torch.cuda.is_available():
        return "cpu"
    dev_idx = torch.cuda.current_device() if device is None or device.type != "cuda" else (device.index or 0)
    props = torch.cuda.get_device_properties(dev_idx)
    clean_name = props.name.replace("NVIDIA ", "").replace("GeForce ", "").replace(" ", "_").replace("-", "_")
    return f"{clean_name}_sm{props.major}{props.minor}"


def get_trt_cache_dir() -> pathlib.Path:
    """Resolve directory where serialized .engine files are cached."""
    # Priority 1: ComfyUI models/dlss/tensorrt
    try:
        import folder_paths
        models_dir = getattr(folder_paths, "models_dir", None)
        if models_dir:
            p = pathlib.Path(models_dir) / "dlss" / "tensorrt"
            p.mkdir(parents=True, exist_ok=True)
            return p
    except Exception:
        pass

    # Priority 2: Home directory fallback
    home = pathlib.Path.home()
    p = home / "ComfyUI" / "models" / "dlss" / "tensorrt"
    p.mkdir(parents=True, exist_ok=True)
    return p


class TensorRTEngineRunner:
    """Executes a deserialized TensorRT engine using zero-copy PyTorch CUDA pointers."""

    def __init__(self, engine_path: str | pathlib.Path, device: torch.device):
        import tensorrt as trt

        self.engine_path = str(engine_path)
        self.device = device
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)

        with open(self.engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine from {self.engine_path}")

        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError(f"Failed to create execution context for engine {self.engine_path}")

        self.num_profiles = self.engine.num_optimization_profiles

    def select_profile_for_shape(self, input_shape: Tuple[int, ...]) -> int:
        """Find the best optimization profile matching the input shape."""
        if self.num_profiles <= 1:
            return 0
        h, w = input_shape[-2], input_shape[-1]
        mp = (h * w) / 1_000_000.0

        if mp <= 1.2:
            return 0  # 1 MP profile
        elif mp <= 1.7:
            return min(1, self.num_profiles - 1)  # 1.5 MP profile
        else:
            return min(2, self.num_profiles - 1)  # 2.0 MP profile

    def __call__(
        self,
        input_tensor: torch.Tensor,
        input_name: str = "input",
        output_name: str = "output",
    ) -> torch.Tensor:
        """Run zero-copy inference directly on a GPU PyTorch tensor."""
        if not input_tensor.is_cuda:
            input_tensor = input_tensor.to(self.device)

        profile_idx = self.select_profile_for_shape(tuple(input_tensor.shape))
        if self.context.active_optimization_profile != profile_idx:
            self.context.set_optimization_profile_async(profile_idx, torch.cuda.current_stream().cuda_stream)

        # Set input tensor dimensions and pointer
        self.context.set_input_shape(input_name, tuple(input_tensor.shape))
        self.context.set_tensor_address(input_name, input_tensor.data_ptr())

        # Determine output shape
        out_shape = tuple(self.context.get_tensor_shape(output_name))
        out_dtype_trt = self.engine.get_tensor_dtype(output_name)
        
        import tensorrt as trt
        torch_dtype = torch.float16 if out_dtype_trt == trt.DataType.HALF else torch.float32

        output_tensor = torch.empty(out_shape, device=self.device, dtype=torch_dtype)
        self.context.set_tensor_address(output_name, output_tensor.data_ptr())

        # Execute on current PyTorch CUDA stream
        stream_ptr = torch.cuda.current_stream().cuda_stream
        success = self.context.execute_async_v3(stream_ptr)
        if not success:
            raise RuntimeError(f"TensorRT execution failed on engine {self.engine_path}")

        return output_tensor


def build_vsr_engine(
    weights_path: str | pathlib.Path,
    engine_path: str | pathlib.Path,
    device: torch.device,
) -> str:
    """Exports VSRModel to ONNX and builds a multi-profile TensorRT engine."""
    import tensorrt as trt
    from safetensors.torch import load_file
    from .vsr import VSRModel

    engine_path = pathlib.Path(engine_path)
    engine_path.parent.mkdir(parents=True, exist_ok=True)

    weights = load_file(str(weights_path))
    model = VSRModel().eval().to(device=device, dtype=torch.float16)
    model.load_safetensors_weights(weights)

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_path = os.path.join(tmpdir, "vsr.onnx")
        dummy_in = torch.randn(1, 16, 256, 256, device=device, dtype=torch.float16)

        # Export to ONNX with dynamic B, H, W
        torch.onnx.export(
            model,
            dummy_in,
            onnx_path,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch", 2: "height", 3: "width"},
                "output": {0: "batch", 2: "height", 3: "width"},
            },
            opset_version=18,
            do_constant_folding=True,
            dynamo=False,
        )

        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
                raise RuntimeError(f"Failed to parse VSR ONNX: {errors}")

    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.FP16)

    # Profile 0: 1.0 MP tier (input H//2, W//2: ~256x256 to 672x384)
    p0 = builder.create_optimization_profile()
    p0.set_shape("input", min=(1, 16, 128, 128), opt=(1, 16, 512, 512), max=(4, 16, 672, 672))
    config.add_optimization_profile(p0)

    # Profile 1: 1.5 MP tier (input H//2, W//2: ~640x640 to 800x800)
    p1 = builder.create_optimization_profile()
    p1.set_shape("input", min=(1, 16, 256, 256), opt=(1, 16, 640, 640), max=(4, 16, 800, 800))
    config.add_optimization_profile(p1)

    # Profile 2: 2.0 MP tier (input H//2, W//2: ~720x720 to 1024x1024)
    p2 = builder.create_optimization_profile()
    p2.set_shape("input", min=(1, 16, 256, 256), opt=(1, 16, 960, 540), max=(4, 16, 1024, 1024))
    config.add_optimization_profile(p2)

    # Build and serialize engine
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("TensorRT failed to build serialized network for VSR")

    with open(str(engine_path), "wb") as f:
        f.write(plan)

    return str(engine_path)


def get_or_build_vsr_trt_runner(
    weights_path: str | pathlib.Path,
    device: torch.device,
) -> TensorRTEngineRunner:
    """Retrieve cached TensorRT runner for VSR or build and cache to disk."""
    if not is_tensorrt_available():
        raise RuntimeError("TensorRT is not available or not installed in the current environment.")

    dev_tag = get_device_tag(device)
    cache_key = f"vsr_{dev_tag}"

    if cache_key in _TRT_RUNNER_CACHE:
        return _TRT_RUNNER_CACHE[cache_key]

    cache_dir = get_trt_cache_dir()
    engine_file = cache_dir / f"vsr_{dev_tag}_fp16.engine"

    if not engine_file.is_file():
        build_vsr_engine(weights_path, engine_file, device)

    runner = TensorRTEngineRunner(engine_file, device=device)
    _TRT_RUNNER_CACHE[cache_key] = runner
    return runner
