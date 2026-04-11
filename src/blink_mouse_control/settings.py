"""Persistent runtime settings for Blink Mouse Control."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class RuntimeSettings:
    """Persisted values that make repeated runs faster and more convenient."""

    calibration_threshold: float
    camera_index: int
    process_size: tuple[int, int]
    camera_size: tuple[int, int]


def get_settings_path() -> Path:
    """Return the path used to store per-user runtime settings."""
    settings_dir = Path.home() / ".blink_mouse_control"
    settings_dir.mkdir(parents=True, exist_ok=True)
    return settings_dir / "settings.json"


def load_runtime_settings() -> RuntimeSettings | None:
    """Load previously saved runtime settings if they exist."""
    settings_path = get_settings_path()
    if not settings_path.exists():
        return None

    try:
        raw_data = json.loads(settings_path.read_text(encoding="utf-8"))
        return RuntimeSettings(
            calibration_threshold=float(raw_data["calibration_threshold"]),
            camera_index=int(raw_data["camera_index"]),
            process_size=tuple(raw_data["process_size"]),
            camera_size=tuple(raw_data["camera_size"]),
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def save_runtime_settings(settings: RuntimeSettings) -> None:
    """Persist runtime settings for faster future launches."""
    settings_path = get_settings_path()
    payload = asdict(settings)
    settings_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
