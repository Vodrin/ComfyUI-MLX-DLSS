"""Video-only correspondence checks; these are not recovered model inputs."""
from dataclasses import dataclass

import numpy as np


@dataclass
class MotionEstimate:
    motion_uv: np.ndarray
    confidence: np.ndarray | None
    reset: bool
    reset_reason: str | None = None


def validate_map(value, height: int, width: int, channels: int, name: str, *, unit_interval=False):
    array = np.asarray(value, dtype=np.float32)
    if array.shape != (height, width, channels) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite with shape ({height}, {width}, {channels})")
    if unit_interval and ((array < 0).any() or (array > 1).any()):
        raise ValueError(f"{name} must be within [0, 1]")
    return array


def luma(frame):
    return frame[..., 0] * 0.2126 + frame[..., 1] * 0.7152 + frame[..., 2] * 0.0722


def assess_motion(current, previous, backward_uv, forward_uv, *, scene_cut_threshold: float) -> MotionEstimate:
    import cv2

    current = np.asarray(current, np.float32)
    if current.ndim != 3 or current.shape[2] != 3:
        raise ValueError("current frame must be HWC RGB")
    height, width = current.shape[:2]
    current = validate_map(current, height, width, 3, "current")
    previous = validate_map(previous, height, width, 3, "previous")
    b = validate_map(backward_uv, height, width, 2, "backward motion")
    f = validate_map(forward_uv, height, width, 2, "forward motion")
    scale = np.array([width, height], np.float32)
    pixels = b * scale
    yy, xx = np.indices((height, width), dtype=np.float32)
    mx, my = xx + pixels[..., 0], yy + pixels[..., 1]
    inside = (mx >= 0) & (mx <= width - 1) & (my >= 0) & (my <= height - 1)
    reverse = cv2.remap(f, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE) * scale
    warped = cv2.remap(previous, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    fb_squared = np.square(pixels + reverse).sum(-1)
    tolerance = 0.01 * (np.square(pixels).sum(-1) + np.square(reverse).sum(-1)) + 0.5
    error = np.abs(current - warped).mean(-1)
    photometric = np.clip((0.12 - error) / 0.09, 0, 1)
    valid = inside & (fb_squared <= tolerance) & (photometric > 0)
    # Keep newly exposed boundaries out of the history filter's footprint.
    valid = cv2.erode(valid.astype(np.uint8), np.ones((7, 7), np.uint8), borderType=cv2.BORDER_REPLICATE)
    confidence = (valid * photometric).astype(np.float32)[..., None]
    cut = False
    reason = None
    if scene_cut_threshold > 0:
        coverage = float((confidence > 0.5).mean())
        warped_error = float(np.abs(luma(current) - luma(warped)).mean())
        # Camera motion can change every pixel without changing the scene.
        if coverage < 0.5 and warped_error > scene_cut_threshold:
            cut, reason = True, "luma change"
        elif coverage < 0.15 and warped_error > 0.12:
            cut, reason = True, "lost correspondence"
    return MotionEstimate(b, confidence, cut, reason)
