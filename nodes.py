"""ComfyUI custom nodes for DLSS5 Neural Rendering, Frame Generation, and RTX VSR."""
from __future__ import annotations

import os
from typing import Tuple
import numpy as np
import torch

try:
    import comfy.model_management
    import comfy.utils
    import folder_paths
except ImportError:
    comfy = None
    folder_paths = None

try:
    from .mlxdlss_core.pipeline import PROFILES
    from .mlxdlss_core.temporal import TemporalOptions, TemporalSession
    from .mlxdlss_core.model_manager import (
        get_neural_rendering_pipeline,
        get_frame_generator,
        get_vsr_resolver,
    )
except (ImportError, ValueError):
    from mlxdlss_core.pipeline import PROFILES
    from mlxdlss_core.temporal import TemporalOptions, TemporalSession
    from mlxdlss_core.model_manager import (
        get_neural_rendering_pipeline,
        get_frame_generator,
        get_vsr_resolver,
    )

# Register "dlss" models folder with ComfyUI
if folder_paths is not None:
    models_dir = folder_paths.models_dir
    dlss_models_dir = os.path.join(models_dir, "dlss")
    os.makedirs(dlss_models_dir, exist_ok=True)
    folder_paths.add_model_folder_path("dlss", dlss_models_dir)


def get_weights_list(default_name: str) -> list[str]:
    files = []
    if folder_paths is not None:
        try:
            files = folder_paths.get_filename_list("dlss")
        except Exception:
            files = []
    if default_name not in files:
        files.insert(0, default_name)
    return files


def get_device() -> torch.device:
    if comfy is not None:
        return comfy.model_management.get_torch_device()
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class DLSS5ImageNode:
    """ComfyUI node for DLSS5 Neural Rendering on images with optional RTX VSR 2x upscaling."""

    @classmethod
    def INPUT_TYPES(cls):
        nr_files = get_weights_list("dlssnr-weights-logical.safetensors")
        vsr_files = get_weights_list("vsr.safetensors")

        return {
            "required": {
                "image": ("IMAGE",),
                "profile": (list(PROFILES.keys()), {"default": "standard"}),
                "intensity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05, "round": 0.001}),
                "detail_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 8.0, "step": 0.1, "round": 0.01}),
                "colour_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.1, "round": 0.01}),
                "detail_radius": ("FLOAT", {"default": 4.0, "min": 0.5, "max": 64.0, "step": 0.5, "round": 0.1}),
                "processing_scale": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 4.0, "step": 0.1, "round": 0.01}),
                "enable_vsr": ("BOOLEAN", {"default": False, "tooltip": "Upscale image 2x using RTX Video Super Resolution"}),
                "precision": (["fast", "reference"], {"default": "fast"}),
                "nr_weights": (nr_files, {"default": "dlssnr-weights-logical.safetensors"}),
                "vsr_weights": (vsr_files, {"default": "vsr.safetensors"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "process_image"
    CATEGORY = "DLSS"

    def process_image(
        self,
        image: torch.Tensor,
        profile: str,
        intensity: float,
        detail_strength: float,
        colour_strength: float,
        detail_radius: float,
        processing_scale: float,
        enable_vsr: bool,
        precision: str,
        nr_weights: str,
        vsr_weights: str,
    ) -> Tuple[torch.Tensor]:
        device = get_device()
        batch_size = image.shape[0]

        # 1. DLSS5 Neural Rendering
        pipeline = None
        if intensity > 0:
            pipeline = get_neural_rendering_pipeline(nr_weights, device=device, precision=precision)

        processed_images = []
        pbar = None
        if comfy is not None:
            total_steps = batch_size * (2 if enable_vsr and intensity > 0 else 1)
            pbar = comfy.utils.ProgressBar(total_steps)

        for i in range(batch_size):
            if comfy is not None:
                comfy.model_management.throw_exception_if_processing_interrupted()

            frame_tensor = image[i]
            frame_np = frame_tensor.cpu().numpy().astype(np.float32)

            if pipeline is not None and intensity > 0:
                result = pipeline.enhance(
                    frame_np,
                    profile=profile,
                    processing_scale=processing_scale,
                    detail_strength=detail_strength,
                    colour_strength=colour_strength,
                    detail_radius=detail_radius,
                    intensity=intensity,
                    frame_index=i,
                )
                frame_out = np.clip(result.image, 0.0, 1.0)
            else:
                frame_out = frame_np

            processed_images.append(frame_out)
            if pbar is not None:
                pbar.update(1)

        output_tensor = torch.from_numpy(np.stack(processed_images, axis=0)).to(torch.float32)

        # 2. Optional RTX VSR 2x Upscaling
        if enable_vsr:
            vsr = get_vsr_resolver(vsr_weights, device=device, precision=precision)
            upscaled_list = []
            for i in range(output_tensor.shape[0]):
                if comfy is not None:
                    comfy.model_management.throw_exception_if_processing_interrupted()

                single_frame = output_tensor[i : i + 1]  # [1, H, W, 3]
                up = vsr.upscale(single_frame)  # [1, 3, 2H, 2W]
                up = up.permute(0, 2, 3, 1).to(torch.float32).cpu()  # [1, 2H, 2W, 3]
                upscaled_list.append(up)
                if pbar is not None:
                    pbar.update(1)
            output_tensor = torch.cat(upscaled_list, dim=0)

        return (output_tensor.clamp(0.0, 1.0),)


class DLSS5VideoNode:
    """ComfyUI node for DLSS5 Neural Rendering on video with Temporal stability, Frame Generation, and RTX VSR."""

    @classmethod
    def INPUT_TYPES(cls):
        nr_files = get_weights_list("dlssnr-weights-logical.safetensors")
        fg_files = get_weights_list("framegen.safetensors")
        vsr_files = get_weights_list("vsr.safetensors")

        return {
            "required": {
                "images": ("IMAGE",),
                "profile": (list(PROFILES.keys()), {"default": "standard"}),
                "intensity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05, "round": 0.001}),
                "detail_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 8.0, "step": 0.1, "round": 0.01}),
                "colour_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.1, "round": 0.01}),
                "detail_radius": ("FLOAT", {"default": 4.0, "min": 0.5, "max": 64.0, "step": 0.5, "round": 0.1}),
                "processing_scale": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 4.0, "step": 0.1, "round": 0.01}),
                "temporal": ("BOOLEAN", {"default": True, "tooltip": "Keep generated detail stable using motion and history"}),
                "motion": (["flow", "zero"], {"default": "flow"}),
                "scene_cut_threshold": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001}),
                "enable_framegen": ("BOOLEAN", {"default": False, "tooltip": "Generate intermediate frames with DLSS Frame Generation"}),
                "framegen_factor": ([2, 3, 4, 8, 16], {"default": 2}),
                "framegen_order": (["nr_first", "fg_first"], {"default": "nr_first"}),
                "enable_vsr": ("BOOLEAN", {"default": False, "tooltip": "Upscale video 2x using RTX Video Super Resolution"}),
                "precision": (["fast", "reference"], {"default": "fast"}),
                "nr_weights": (nr_files, {"default": "dlssnr-weights-logical.safetensors"}),
                "framegen_weights": (fg_files, {"default": "framegen.safetensors"}),
                "vsr_weights": (vsr_files, {"default": "vsr.safetensors"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "process_video"
    CATEGORY = "DLSS"

    def _apply_neural_rendering(
        self,
        frames_np: list[np.ndarray],
        pipeline,
        temporal: bool,
        profile: str,
        intensity: float,
        detail_strength: float,
        colour_strength: float,
        detail_radius: float,
        processing_scale: float,
        motion: str,
        scene_cut_threshold: float,
        pbar=None,
    ) -> list[np.ndarray]:
        if intensity <= 0 or pipeline is None:
            return frames_np

        output_frames = []
        if temporal:
            options = TemporalOptions(
                processing_scale=processing_scale,
                profile=profile,
                intensity=intensity,
                detail_strength=detail_strength,
                colour_strength=colour_strength,
                detail_radius=detail_radius,
                scene_cut_threshold=scene_cut_threshold,
            )
            session = TemporalSession(pipeline, options=options, motion=motion)
            for idx, frame in enumerate(frames_np):
                if comfy is not None:
                    comfy.model_management.throw_exception_if_processing_interrupted()
                res = session.process(frame)
                output_frames.append(np.clip(res, 0.0, 1.0))
                if pbar is not None:
                    pbar.update(1)
        else:
            for idx, frame in enumerate(frames_np):
                if comfy is not None:
                    comfy.model_management.throw_exception_if_processing_interrupted()
                res = pipeline.enhance(
                    frame,
                    profile=profile,
                    processing_scale=processing_scale,
                    detail_strength=detail_strength,
                    colour_strength=colour_strength,
                    detail_radius=detail_radius,
                    intensity=intensity,
                    frame_index=idx,
                )
                output_frames.append(np.clip(res.image, 0.0, 1.0))
                if pbar is not None:
                    pbar.update(1)

        return output_frames

    def _apply_framegen(
        self,
        frames_np: list[np.ndarray],
        generator,
        factor: int,
        pbar=None,
    ) -> list[np.ndarray]:
        if len(frames_np) < 2 or factor < 2:
            return frames_np

        generated_pairs = generator.generate_pairs(frames_np, factor=factor, as_uint8=False)
        interleaved = [frames_np[0]]
        for i in range(len(frames_np) - 1):
            pair_gen = generated_pairs[i]
            interleaved.extend(pair_gen)
            interleaved.append(frames_np[i + 1])
            if pbar is not None:
                pbar.update(1)

        return [np.clip(f, 0.0, 1.0) for f in interleaved]

    def _apply_vsr(
        self,
        frames_np: list[np.ndarray],
        vsr,
        pbar=None,
    ) -> list[np.ndarray]:
        out_frames = []
        for frame in frames_np:
            if comfy is not None:
                comfy.model_management.throw_exception_if_processing_interrupted()
            tensor = torch.from_numpy(np.ascontiguousarray(frame)).permute(2, 0, 1)[None]  # [1, 3, H, W]
            up = vsr.upscale(tensor)  # [1, 3, 2H, 2W]
            up_np = up.squeeze(0).permute(1, 2, 0).to(torch.float32).cpu().numpy()  # [2H, 2W, 3]
            out_frames.append(np.clip(up_np, 0.0, 1.0))
            if pbar is not None:
                pbar.update(1)
        return out_frames

    def process_video(
        self,
        images: torch.Tensor,
        profile: str,
        intensity: float,
        detail_strength: float,
        colour_strength: float,
        detail_radius: float,
        processing_scale: float,
        temporal: bool,
        motion: str,
        scene_cut_threshold: float,
        enable_framegen: bool,
        framegen_factor: int,
        framegen_order: str,
        enable_vsr: bool,
        precision: str,
        nr_weights: str,
        framegen_weights: str,
        vsr_weights: str,
    ) -> Tuple[torch.Tensor]:
        device = get_device()
        num_frames = images.shape[0]

        # Convert input tensor batch to list of numpy float32 arrays
        frames_np = [images[i].cpu().numpy().astype(np.float32) for i in range(num_frames)]

        # Estimate total progress bar steps
        total_steps = 0
        if intensity > 0:
            total_steps += num_frames
        if enable_framegen and num_frames >= 2:
            total_steps += num_frames - 1
            num_after_fg = 1 + (num_frames - 1) * framegen_factor
        else:
            num_after_fg = num_frames

        if enable_vsr:
            total_steps += num_after_fg

        pbar = None
        if comfy is not None:
            pbar = comfy.utils.ProgressBar(max(1, total_steps))

        # Pipelines
        pipeline = None
        if intensity > 0:
            pipeline = get_neural_rendering_pipeline(nr_weights, device=device, precision=precision)

        generator = None
        if enable_framegen and num_frames >= 2:
            generator = get_frame_generator(framegen_weights, device=device, precision=precision)

        # Execution order
        current_frames = frames_np
        if enable_framegen and framegen_order == "fg_first" and generator is not None:
            current_frames = self._apply_framegen(current_frames, generator, factor=framegen_factor, pbar=pbar)
            current_frames = self._apply_neural_rendering(
                current_frames,
                pipeline,
                temporal=temporal,
                profile=profile,
                intensity=intensity,
                detail_strength=detail_strength,
                colour_strength=colour_strength,
                detail_radius=detail_radius,
                processing_scale=processing_scale,
                motion=motion,
                scene_cut_threshold=scene_cut_threshold,
                pbar=pbar,
            )
        else:
            # nr_first (default)
            current_frames = self._apply_neural_rendering(
                current_frames,
                pipeline,
                temporal=temporal,
                profile=profile,
                intensity=intensity,
                detail_strength=detail_strength,
                colour_strength=colour_strength,
                detail_radius=detail_radius,
                processing_scale=processing_scale,
                motion=motion,
                scene_cut_threshold=scene_cut_threshold,
                pbar=pbar,
            )
            if enable_framegen and generator is not None:
                current_frames = self._apply_framegen(current_frames, generator, factor=framegen_factor, pbar=pbar)

        if enable_vsr:
            vsr = get_vsr_resolver(vsr_weights, device=device, precision=precision)
            current_frames = self._apply_vsr(current_frames, vsr, pbar=pbar)

        output_tensor = torch.from_numpy(np.stack(current_frames, axis=0)).to(torch.float32)
        return (output_tensor.clamp(0.0, 1.0),)
