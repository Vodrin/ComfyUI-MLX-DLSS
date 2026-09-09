"""ComfyUI MLX-DLSS Custom Node Package.

Provides DLSS5 Neural Rendering, DLSS Frame Generation, and RTX Video Super Resolution (VSR)
ported from https://github.com/iamwavecut/MLX-DLSS.
"""
from .nodes import DLSS5ImageNode, DLSS5VideoNode

NODE_CLASS_MAPPINGS = {
    "DLSS5ImageNode": DLSS5ImageNode,
    "DLSS5VideoNode": DLSS5VideoNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DLSS5ImageNode": "DLSS5 Neural Rendering (Image)",
    "DLSS5VideoNode": "DLSS5 Neural Rendering (Video)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
