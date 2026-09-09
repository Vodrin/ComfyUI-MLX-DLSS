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

        self.num_profiles = self.engine.num_optimization_profiles
        self.stream = torch.cuda.Stream(device=self.device)
        stream_ptr = self.stream.cuda_stream
        self.contexts = []
        for i in range(self.num_profiles):
            ctx = self.engine.create_execution_context()
            if ctx is None:
                raise RuntimeError(f"Failed to create execution context {i} for engine {self.engine_path}")
            ctx.set_optimization_profile_async(i, stream_ptr)
            self.contexts.append(ctx)

        # Inspect I/O tensors
        self.input_names = []
        self.output_names = []
        for i in range(self.engine.num_io_tensors):
            t_name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(t_name) == trt.TensorIOMode.INPUT:
                self.input_names.append(t_name)
            else:
                self.output_names.append(t_name)

        # Discover profile dimensions for first input
        self.profile_bounds = []
        if self.input_names:
            first_in = self.input_names[0]
            for i in range(self.num_profiles):
                self.profile_bounds.append(self.engine.get_tensor_profile_shape(first_in, i))

    def select_profile_for_shape(self, input_shape: Tuple[int, ...]) -> int:
        """Find the best optimization profile matching the input shape."""
        if self.num_profiles <= 1:
            return 0

        # Try matching against profile bounds dynamically
        if self.profile_bounds:
            for i, (min_s, opt_s, max_s) in enumerate(self.profile_bounds):
                if len(input_shape) == len(max_s) and all(s <= m for s, m in zip(input_shape, max_s)) and all(s >= mn for s, mn in zip(input_shape, min_s)):
                    return i
            raise ValueError(
                f"Input shape {input_shape} does not fit within any TensorRT profile bounds (bounds: {self.profile_bounds})"
            )

        # Fallback to dimension heuristic
        h, w = input_shape[-2], input_shape[-1]
        max_dim = max(h, w)
        if max_dim <= 672:
            return 0
        elif max_dim <= 800:
            return min(1, self.num_profiles - 1)
        else:
            return min(2, self.num_profiles - 1)

    def __call__(
        self,
        inputs: torch.Tensor | dict[str, torch.Tensor],
        output_name: str | list[str] | None = None,
        input_name: str | None = None,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Run zero-copy inference directly on GPU PyTorch tensor(s)."""
        import tensorrt as trt

        if isinstance(inputs, torch.Tensor):
            in_name = input_name or (self.input_names[0] if self.input_names else "input")
            inputs_dict = {in_name: inputs}
        else:
            inputs_dict = inputs

        first_tensor = next(iter(inputs_dict.values()))
        profile_idx = self.select_profile_for_shape(tuple(first_tensor.shape))
        context = self.contexts[profile_idx]

        # Bind all input tensors
        for name, tensor in inputs_dict.items():
            if not tensor.is_cuda:
                tensor = tensor.to(self.device)
            success = context.set_input_shape(name, tuple(tensor.shape))
            if not success:
                raise ValueError(
                    f"TensorRT failed to set input shape {tuple(tensor.shape)} on profile {profile_idx}"
                )
            context.set_tensor_address(name, tensor.data_ptr())

        # Determine output tensors to allocate and bind
        if output_name is None:
            target_outputs = self.output_names
        elif isinstance(output_name, str):
            target_outputs = [output_name]
        else:
            target_outputs = list(output_name)

        output_tensors = {}
        for out_name in target_outputs:
            out_shape = tuple(context.get_tensor_shape(out_name))
            if any(d <= 0 for d in out_shape):
                raise ValueError(
                    f"TensorRT returned invalid output shape {out_shape} for tensor '{out_name}'"
                )
            out_dtype_trt = self.engine.get_tensor_dtype(out_name)
            torch_dtype = torch.float16 if out_dtype_trt == trt.DataType.HALF else torch.float32
            out_tensor = torch.empty(out_shape, device=self.device, dtype=torch_dtype)
            context.set_tensor_address(out_name, out_tensor.data_ptr())
            output_tensors[out_name] = out_tensor

        # Execute on dedicated CUDA stream
        with torch.cuda.stream(self.stream):
            stream_ptr = self.stream.cuda_stream
            success = context.execute_async_v3(stream_ptr)
            if not success:
                raise RuntimeError(f"TensorRT execution failed on engine {self.engine_path} (profile {profile_idx})")
        torch.cuda.current_stream().wait_stream(self.stream)

        if isinstance(output_name, str) or (output_name is None and len(self.output_names) == 1):
            return output_tensors[target_outputs[0]]
        return output_tensors


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
    network = builder.create_network()
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

    # Profile 0: 1.0 MP tier (input H//2, W//2: ~256x256 to 672x384)
    p0 = builder.create_optimization_profile()
    p0.set_shape("input", min=(1, 16, 128, 128), opt=(1, 16, 512, 512), max=(4, 16, 672, 672))
    config.add_optimization_profile(p0)

    # Profile 1: 1.5 MP tier (input H//2, W//2: ~640x640 to 800x800)
    p1 = builder.create_optimization_profile()
    p1.set_shape("input", min=(1, 16, 256, 256), opt=(1, 16, 640, 640), max=(4, 16, 800, 800))
    config.add_optimization_profile(p1)

    # Profile 2: 2.0 MP tier (input H//2, W//2: 1080p padded to 544x960 up to 1024x1024)
    p2 = builder.create_optimization_profile()
    p2.set_shape("input", min=(1, 16, 256, 256), opt=(1, 16, 544, 960), max=(4, 16, 1024, 1024))
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


def _warp_onnx(img: torch.Tensor, fx: torch.Tensor, fy: torch.Tensor) -> torch.Tensor:
    h = img.shape[-2]
    w = img.shape[-1]
    ys = torch.arange(h, device=img.device, dtype=torch.float32).view(-1, 1)
    xs = torch.arange(w, device=img.device, dtype=torch.float32).view(1, -1)
    w_scale = 2.0 / torch.clamp((torch.tensor(w, device=img.device, dtype=torch.float32) - 1.0), min=1.0)
    h_scale = 2.0 / torch.clamp((torch.tensor(h, device=img.device, dtype=torch.float32) - 1.0), min=1.0)
    gx = (xs + fx.float()) * w_scale - 1.0
    gy = (ys + fy.float()) * h_scale - 1.0
    grid = torch.stack([gx, gy], -1).to(img.dtype)
    return F.grid_sample(img, grid, mode="bilinear", padding_mode="border", align_corners=True)


class FrameGenNet(torch.nn.Module):
    def __init__(self, fg_inst):
        super().__init__()
        self.block0 = fg_inst.block0
        self.block1 = fg_inst.block1

    def forward(self, a: torch.Tensor, b: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        from .framegen import photometric_error, _up2

        err = photometric_error(a, b)
        zero = torch.zeros_like(err)
        cand_a = torch.cat([a, err], 1)
        cand_b = torch.cat([b, err], 1)
        f0, m0, r0 = self.block0(torch.cat([cand_a, cand_b, zero, phase], 1))
        f0, m0, r0 = _up2(f0), _up2(m0), _up2(r0)
        warped_a = _warp_onnx(cand_a, 2 * f0[:, 0], 2 * f0[:, 1])
        warped_b = _warp_onnx(cand_b, 2 * f0[:, 2], 2 * f0[:, 3])
        f1, m1, r1 = self.block1(torch.cat([warped_a, warped_b, f0, m0, zero, r0, phase], 1))
        return torch.cat([2 * f0 + f1, m0 + m1, r0 + r1], 1)


def build_framegen_engine(
    weights_path_or_dict: str | pathlib.Path | dict[str, torch.Tensor],
    engine_path: str | pathlib.Path,
    device: torch.device,
) -> str:
    """Exports FrameGenNet to ONNX and compiles a multi-profile TensorRT engine."""
    import tensorrt as trt
    from safetensors.torch import load_file
    from .framegen import FrameGenerator

    engine_path = pathlib.Path(engine_path)
    engine_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(weights_path_or_dict, (str, pathlib.Path)):
        weights = load_file(str(weights_path_or_dict))
    else:
        weights = weights_path_or_dict

    fg = FrameGenerator(weights, device=device, precision="fast")
    model = FrameGenNet(fg).eval().half()

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network()
    parser = trt.OnnxParser(network, logger)

    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_path = os.path.join(tmpdir, "framegen.onnx")
        dummy_a = torch.randn(1, 3, 256, 256, device=device, dtype=torch.float16)
        dummy_b = torch.randn(1, 3, 256, 256, device=device, dtype=torch.float16)
        dummy_p = torch.full((1, 1, 256, 256), 0.5, device=device, dtype=torch.float16)

        torch.onnx.export(
            model,
            (dummy_a, dummy_b, dummy_p),
            onnx_path,
            input_names=["a", "b", "phase"],
            output_names=["export"],
            dynamic_axes={
                "a": {0: "batch", 2: "height", 3: "width"},
                "b": {0: "batch", 2: "height", 3: "width"},
                "phase": {0: "batch", 2: "height", 3: "width"},
                "export": {0: "batch", 2: "height", 3: "width"},
            },
            opset_version=17,
            do_constant_folding=True,
            dynamo=False,
        )

        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
                raise RuntimeError(f"Failed to parse FrameGen ONNX: {errors}")

    config = builder.create_builder_config()

    # Profile 0: 1.0 MP tier (input H//2, W//2: ~128x128 to 384x384)
    p0 = builder.create_optimization_profile()
    p0.set_shape("a", min=(1, 3, 128, 128), opt=(1, 3, 256, 256), max=(8, 3, 384, 384))
    p0.set_shape("b", min=(1, 3, 128, 128), opt=(1, 3, 256, 256), max=(8, 3, 384, 384))
    p0.set_shape("phase", min=(1, 1, 128, 128), opt=(1, 1, 256, 256), max=(8, 1, 384, 384))
    config.add_optimization_profile(p0)

    # Profile 1: 1.5 MP tier (input H//2, W//2: ~256x256 to 512x512)
    p1 = builder.create_optimization_profile()
    p1.set_shape("a", min=(1, 3, 128, 128), opt=(1, 3, 384, 384), max=(8, 3, 512, 512))
    p1.set_shape("b", min=(1, 3, 128, 128), opt=(1, 3, 384, 384), max=(8, 3, 512, 512))
    p1.set_shape("phase", min=(1, 1, 128, 128), opt=(1, 1, 384, 384), max=(8, 1, 512, 512))
    config.add_optimization_profile(p1)

    # Profile 2: 2.0 MP tier (1080p padded is 544x960, up to 768x1024)
    p2 = builder.create_optimization_profile()
    p2.set_shape("a", min=(1, 3, 128, 128), opt=(1, 3, 544, 960), max=(8, 3, 768, 1024))
    p2.set_shape("b", min=(1, 3, 128, 128), opt=(1, 3, 544, 960), max=(8, 3, 768, 1024))
    p2.set_shape("phase", min=(1, 1, 128, 128), opt=(1, 1, 544, 960), max=(8, 1, 768, 1024))
    config.add_optimization_profile(p2)

    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("TensorRT failed to build serialized network for FrameGen")

    with open(str(engine_path), "wb") as f:
        f.write(plan)

    return str(engine_path)


def get_or_build_framegen_trt_runner(
    weights_path_or_dict: str | pathlib.Path | dict[str, torch.Tensor],
    device: torch.device,
) -> TensorRTEngineRunner:
    """Retrieve cached TensorRT runner for FrameGen or build and cache to disk."""
    if not is_tensorrt_available():
        raise RuntimeError("TensorRT is not available or not installed in the current environment.")

    dev_tag = get_device_tag(device)
    cache_key = f"framegen_{dev_tag}"

    if cache_key in _TRT_RUNNER_CACHE:
        return _TRT_RUNNER_CACHE[cache_key]

    cache_dir = get_trt_cache_dir()
    engine_file = cache_dir / f"framegen_{dev_tag}_fp16.engine"

    if not engine_file.is_file():
        build_framegen_engine(weights_path_or_dict, engine_file, device)

    runner = TensorRTEngineRunner(engine_file, device=device)
    _TRT_RUNNER_CACHE[cache_key] = runner
    return runner


def build_nr_engine(
    weights_path_or_dict: str | pathlib.Path | dict[str, torch.Tensor],
    engine_path: str | pathlib.Path,
    device: torch.device,
) -> str:
    """Exports NeuralRenderingModel to dynamic ONNX and compiles a TensorRT FP16 engine."""
    import tensorrt as trt
    from safetensors.torch import load_file
    from .model import NeuralRenderingModel

    if isinstance(weights_path_or_dict, (str, pathlib.Path)):
        weights = load_file(str(weights_path_or_dict))
    else:
        weights = weights_path_or_dict

    old_chunk = os.environ.get("MLXDLSS_TORCH_CHUNK_TOKENS", None)
    os.environ["MLXDLSS_TORCH_CHUNK_TOKENS"] = "0"
    try:
        model = NeuralRenderingModel(weights).eval().to(device=device, dtype=torch.float16)

        logger = trt.Logger(trt.Logger.INFO)
        builder = trt.Builder(logger)
        network = builder.create_network()
        parser = trt.OnnxParser(network, logger)

        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = os.path.join(tmpdir, "nr.onnx")
            dummy_in = torch.randn(1, 320, 320, 16, device=device, dtype=torch.float16)

            torch.onnx.export(
                model,
                dummy_in,
                onnx_path,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={
                    "input": {0: "batch", 1: "height", 2: "width"},
                    "output": {0: "batch", 1: "height", 2: "width"},
                },
                opset_version=18,
                do_constant_folding=True,
                dynamo=False,
            )

            with open(onnx_path, "rb") as f:
                if not parser.parse(f.read()):
                    errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
                    raise RuntimeError(f"Failed to parse Neural Rendering ONNX: {errors}")
    finally:
        if old_chunk is not None:
            os.environ["MLXDLSS_TORCH_CHUNK_TOKENS"] = old_chunk
        else:
            os.environ.pop("MLXDLSS_TORCH_CHUNK_TOKENS", None)

    config = builder.create_builder_config()
    # Allow TensorRT ample workspace for 71-block transformer activations (requires ~11GB)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 16 * (1024 ** 3))

    # Profile 0: Dynamic tier covering vendor extents (320x320 up to 1024x1024, opt 512x512)
    p0 = builder.create_optimization_profile()
    p0.set_shape("input", min=(1, 320, 320, 16), opt=(1, 512, 512, 16), max=(1, 1024, 1024, 16))
    config.add_optimization_profile(p0)

    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("TensorRT failed to build serialized network for Neural Rendering")

    with open(str(engine_path), "wb") as f:
        f.write(plan)

    return str(engine_path)


def get_or_build_nr_trt_runner(
    weights_path_or_dict: str | pathlib.Path | dict[str, torch.Tensor],
    device: torch.device,
) -> TensorRTEngineRunner:
    """Retrieve cached TensorRT runner for Neural Rendering or build and cache to disk."""
    if not is_tensorrt_available():
        raise RuntimeError("TensorRT is not available or not installed in the current environment.")

    dev_tag = get_device_tag(device)
    cache_key = f"nr_{dev_tag}"

    if cache_key in _TRT_RUNNER_CACHE:
        return _TRT_RUNNER_CACHE[cache_key]

    cache_dir = get_trt_cache_dir()
    engine_file = cache_dir / f"nr_{dev_tag}_fp16.engine"

    if not engine_file.is_file():
        build_nr_engine(weights_path_or_dict, engine_file, device)

    runner = TensorRTEngineRunner(engine_file, device=device)
    _TRT_RUNNER_CACHE[cache_key] = runner
    return runner
