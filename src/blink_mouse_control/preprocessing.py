"""Frame preprocessing helpers for robust face landmark detection."""

from typing import Any

import cv2
import numpy as np


def _compute_equalization_strength(mean_luma: float) -> float:
    """Return blend strength for luminance equalization based on scene brightness."""
    if mean_luma < 70.0:
        return 0.72
    if mean_luma < 105.0:
        return 0.58
    if mean_luma > 190.0:
        return 0.52
    if mean_luma > 165.0:
        return 0.42
    return 0.34


def normalize_lighting(frame: Any, adaptive_adjustment: bool = True) -> Any:
    """Normalize lighting in BGR frames while preserving natural colors.

    Steps:
    1) Equalize only Y (luminance) in YCrCb.
    2) Blend equalized luminance with the original to avoid washed-out results.
    3) Optionally apply a small adaptive brightness/contrast correction from mean luma.
    """
    if frame is None or frame.size == 0:
        return frame

    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)

    mean_luma = float(np.mean(y))
    equalized_y = cv2.equalizeHist(y)
    strength = _compute_equalization_strength(mean_luma)
    blended_y = cv2.addWeighted(y, 1.0 - strength, equalized_y, strength, 0.0)

    if adaptive_adjustment:
        luma_delta = (128.0 - mean_luma) / 128.0
        alpha = float(np.clip(1.0 + (0.16 * luma_delta), 0.92, 1.08))
        beta = float(np.clip(9.0 * luma_delta, -10.0, 10.0))
        blended_y = cv2.convertScaleAbs(blended_y, alpha=alpha, beta=beta)

    normalized_ycrcb = cv2.merge((blended_y, cr, cb))
    return cv2.cvtColor(normalized_ycrcb, cv2.COLOR_YCrCb2BGR)
