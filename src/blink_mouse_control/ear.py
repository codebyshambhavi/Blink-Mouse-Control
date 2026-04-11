"""Eye Aspect Ratio (EAR) utilities."""

from collections.abc import Sequence

import numpy as np


def euclidean(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Return Euclidean distance between two 2D points."""
    return float(np.linalg.norm(np.array(a) - np.array(b)))


def calculate_ear(eye_points: Sequence[int], landmarks: Sequence[object]) -> float:
    """Calculate EAR for one eye using six landmark indices."""
    coords = [(landmarks[i].x, landmarks[i].y) for i in eye_points]
    vertical_1 = euclidean(coords[1], coords[5])
    vertical_2 = euclidean(coords[2], coords[4])
    horizontal = euclidean(coords[0], coords[3])
    if horizontal == 0:
        return 0.0
    return (vertical_1 + vertical_2) / (2.0 * horizontal)


def compute_threshold_from_samples(ears: Sequence[float]) -> float:
    """Compute a stable EAR threshold from calibration samples."""
    sample_array = np.array(ears, dtype=float)
    if len(sample_array) == 0:
        return 0.22

    open_median = float(np.median(sample_array))
    closed_10pct = float(np.percentile(sample_array, 10))
    threshold = float(np.mean([open_median * 0.7, closed_10pct * 1.3]))
    return max(0.12, min(threshold, 0.35))
