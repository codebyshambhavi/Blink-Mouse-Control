"""Model asset management for MediaPipe Face Landmarker."""

from __future__ import annotations

import os
import tempfile
import urllib.request
from pathlib import Path

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)
MODEL_FILENAME = "face_landmarker.task"


def get_model_path() -> Path:
    """Return the cached local path for the Face Landmarker model."""
    cache_dir = Path(os.environ.get("BLINK_MOUSE_CONTROL_CACHE", tempfile.gettempdir()))
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / MODEL_FILENAME


def ensure_model_available() -> Path:
    """Download the Face Landmarker model if it is not already cached."""
    model_path = get_model_path()
    if model_path.exists():
        return model_path

    print("[INFO] Downloading Face Landmarker model...")
    urllib.request.urlretrieve(MODEL_URL, model_path)
    return model_path
