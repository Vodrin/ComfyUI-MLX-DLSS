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


def assess_motion(current, previous, backward_uv, forward_uv, *, scene_cut_threshold: float, device=None) -> MotionEstimate:
    current = np.asarray(current, np.float32)
    if current.ndim != 3 or current.shape[2] != 3:
        raise ValueError("current frame must be HWC RGB")
    height, width = current.shape[:2]
    current = validate_map(current, height, width, 3, "current")
    previous = validate_map(previous, height, width, 3, "previous")
    b = validate_map(backward_uv, height, width, 2, "backward motion")
    f = validate_map(forward_uv, height, width, 2, "forward motion")

    import torch
    import torch.nn.functional as F

    if torch.cuda.is_available():
        dev = device or torch.device("cuda")
        curr_t = torch.from_numpy(current).permute(2, 0, 1)[None].to(dev, torch.float32)
        prev_t = torch.from_numpy(previous).permute(2, 0, 1)[None].to(dev, torch.float32)
        b_t = torch.from_numpy(b).permute(2, 0, 1)[None].to(dev, torch.float32)
        f_t = torch.from_numpy(f).permute(2, 0, 1)[None].to(dev, torch.float32)

        ys, xs = torch.meshgrid(
            torch.arange(height, device=dev, dtype=torch.float32),
            torch.arange(width, device=dev, dtype=torch.float32),
            indexing="ij",
        )
        scale_x = float(width)
        scale_y = float(height)
        pix_x = b_t[0, 0] * scale_x
        pix_y = b_t[0, 1] * scale_y

        mx = xs + pix_x
        my = ys + pix_y
        inside = (mx >= 0.0) & (mx <= width - 1.0) & (my >= 0.0) & (my <= height - 1.0)

        gx = mx * (2.0 / max(width - 1, 1)) - 1.0
        gy = my * (2.0 / max(height - 1, 1)) - 1.0
        grid = torch.stack([gx, gy], dim=-1)[None]

        warped = F.grid_sample(prev_t, grid, mode="bilinear", padding_mode="border", align_corners=True)
        reverse = F.grid_sample(f_t, grid, mode="bilinear", padding_mode="border", align_corners=True)
        rev_x = reverse[0, 0] * scale_x
        rev_y = reverse[0, 1] * scale_y

        fb_squared = (pix_x + rev_x).square() + (pix_y + rev_y).square()
        tolerance = 0.01 * ((pix_x.square() + pix_y.square()) + (rev_x.square() + rev_y.square())) + 0.5
        error = (curr_t - warped).abs().mean(1, keepdim=True)[0, 0]
        photometric = ((0.12 - error) / 0.09).clamp(0.0, 1.0)
        valid = (inside & (fb_squared <= tolerance) & (photometric > 0.0)).to(torch.float32)[None, None]

        valid_eroded = -F.max_pool2d(-valid, 7, stride=1, padding=3)
        confidence = (valid_eroded[0, 0] * photometric)[..., None]

        cut = False
        reason = None
        if scene_cut_threshold > 0:
            coverage = float((confidence > 0.5).to(torch.float32).mean().item())
            curr_luma = curr_t[0, 0] * 0.2126 + curr_t[0, 1] * 0.7152 + curr_t[0, 2] * 0.0722
            warp_luma = warped[0, 0] * 0.2126 + warped[0, 1] * 0.7152 + warped[0, 2] * 0.0722
            warped_error = float((curr_luma - warp_luma).abs().mean().item())
            if coverage < 0.5 and warped_error > scene_cut_threshold:
                cut, reason = True, "luma change"
            elif coverage < 0.15 and warped_error > 0.12:
                cut, reason = True, "lost correspondence"

        return MotionEstimate(b, confidence.cpu().numpy(), cut, reason)

    import cv2

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
