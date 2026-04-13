"""Project configuration and constants."""

from dataclasses import dataclass


BEAUTY_FILTER_LEVELS: tuple[str, ...] = ("Off", "Low", "Medium", "High")


@dataclass(frozen=True)
class DetectionConfig:
    """Runtime settings for blink detection and actions."""

    camera_index: int = 0
    frame_size: tuple[int, int] = (960, 540)
    process_size: tuple[int, int] = (640, 360)
    calibration_preview_size: tuple[int, int] = (640, 360)
    calibration_time_seconds: float = 4.0
    ear_smooth_window: int = 6
    double_blink_max_gap_seconds: float = 0.55
    long_blink_min_duration_seconds: float = 1.0
    click_hold_duration_seconds: float = 0.5
    scroll_pitch_threshold: float = 0.022
    scroll_cooldown_seconds: float = 0.14
    scroll_step_pixels: int = 80
    scroll_pitch_smoothing: float = 0.22
    scroll_neutral_adapt_rate: float = 0.02
    fallback_ear_threshold: float = 0.22
    min_detection_confidence: float = 0.5
    max_num_faces: int = 1
    refine_landmarks: bool = True
    camera_width: int = 640
    camera_height: int = 360
    max_camera_fps: int = 30
    use_saved_calibration: bool = True
    show_help_overlay: bool = True
    save_calibration_after_run: bool = True
    beauty_filter_level: str = "Medium"


LEFT_EYE_LANDMARKS: tuple[int, int, int, int, int, int] = (33, 160, 158, 133, 153, 144)
RIGHT_EYE_LANDMARKS: tuple[int, int, int, int, int, int] = (362, 385, 387, 263, 373, 380)
