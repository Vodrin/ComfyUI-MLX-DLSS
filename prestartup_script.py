"""Prestartup script for ComfyUI-MLX-DLSS.

Ensures TensorRT 11 runtime libraries (libnvinfer.so.11) are loaded into the process
before any third-party nodes that may bundle older TensorRT versions (such as nvvfx with TRT 10).
"""
try:
    import tensorrt as trt
except Exception:
    pass
