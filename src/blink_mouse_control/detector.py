"""Real-time blink detection orchestration."""

import os
import threading
import time
from collections.abc import Callable
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from .actions import MouseActions, NoOpMouseActions
from .calibration import calibrate_ear_threshold
from .config import DetectionConfig, LEFT_EYE_LANDMARKS, RIGHT_EYE_LANDMARKS
from .ear import calculate_ear
from .model import ensure_model_available
from .settings import RuntimeSettings, load_runtime_settings, save_runtime_settings
from .overlay import draw_no_face_overlay, draw_status_overlay


@dataclass
class BlinkState:
    """Mutable state used to infer blink patterns across frames."""

    in_blink: bool = False
    blink_start: float | None = None
    blink_times: list[float] = field(default_factory=list)


@dataclass
class DetectionControl:
    """Thread-safe runtime control shared between UI and detector loop."""

    stop_event: threading.Event = field(default_factory=threading.Event)
    finished_event: threading.Event = field(default_factory=threading.Event)
    recalibrate_event: threading.Event = field(default_factory=threading.Event)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _threshold_override: float | None = None
    _cursor_control_enabled: bool = True
    _blink_count: int = 0
    _current_fps: float = 0.0
    _current_ear: float | None = None
    _current_threshold: float = 0.0

    def __post_init__(self) -> None:
        """Initialize lifecycle flags to a stopped state."""
        self.finished_event.set()

    def mark_started(self) -> None:
        """Mark detector worker as running."""
        self.finished_event.clear()

    def mark_stopped(self) -> None:
        """Mark detector worker as stopped."""
        self.finished_event.set()

    def wait_until_stopped(self, timeout: float | None = None) -> bool:
        """Wait until worker thread signals stop completion."""
        return self.finished_event.wait(timeout=timeout)

    def request_stop(self) -> None:
        self.stop_event.set()

    def should_stop(self) -> bool:
        return self.stop_event.is_set()

    def request_recalibration(self) -> None:
        self.recalibrate_event.set()

    def consume_recalibration_request(self) -> bool:
        if self.recalibrate_event.is_set():
            self.recalibrate_event.clear()
            return True
        return False

    def set_threshold_override(self, value: float | None) -> None:
        with self._lock:
            self._threshold_override = value

    def get_threshold_override(self) -> float | None:
        with self._lock:
            return self._threshold_override

    def set_cursor_control_enabled(self, enabled: bool) -> None:
        with self._lock:
            self._cursor_control_enabled = enabled

    def is_cursor_control_enabled(self) -> bool:
        with self._lock:
            return self._cursor_control_enabled

    def increment_blink_count(self) -> None:
        with self._lock:
            self._blink_count += 1

    def update_live_stats(
        self,
        *,
        fps: float,
        ear: float | None,
        threshold: float,
    ) -> None:
        """Update live runtime statistics shown in the desktop UI."""
        with self._lock:
            self._current_fps = fps
            self._current_ear = ear
            self._current_threshold = threshold

    def get_live_stats(self) -> dict[str, float | int | None]:
        """Get a thread-safe snapshot of current runtime statistics."""
        with self._lock:
            return {
                "fps": self._current_fps,
                "ear": self._current_ear,
                "blink_count": self._blink_count,
                "threshold": self._current_threshold,
            }


def _open_camera(camera_index: int) -> cv2.VideoCapture:
    """Return an opened VideoCapture or an unopened handle if unavailable."""
    backend = cv2.CAP_DSHOW if os.name == "nt" else 0
    return cv2.VideoCapture(camera_index, backend)


def _configure_camera(cap: cv2.VideoCapture, config: DetectionConfig) -> None:
    """Request a smaller camera stream to reduce bandwidth and CPU usage."""
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.camera_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.camera_height)
    cap.set(cv2.CAP_PROP_FPS, config.max_camera_fps)


def _read_frame(cap: cv2.VideoCapture) -> tuple[bool, Any | None]:
    """Read a frame and report capture errors in a single place."""
    try:
        return cap.read()
    except cv2.error as exc:
        print(f"[ERROR] Camera read failed: {exc}")
        return False, None


def _smooth_ear(ear_history: deque[float], current_ear: float) -> float:
    """Append current EAR and return the moving average."""
    ear_history.append(current_ear)
    return sum(ear_history) / len(ear_history)


def _complete_blink_if_needed(
    now: float,
    state: BlinkState,
    actions: MouseActions,
    config: DetectionConfig,
    control: DetectionControl,
) -> None:
    """Finalize a blink and dispatch long-blink action when threshold is met."""
    if state.blink_start is None:
        return

    duration = now - state.blink_start
    state.blink_times.append(now)
    control.increment_blink_count()
    if duration >= config.long_blink_min_duration_seconds:
        print(f"[EVENT] LONG blink ({duration:.2f}s)")
        actions.hold_left_click(config.click_hold_duration_seconds)
        state.blink_times.clear()


def _update_blink_state(
    smooth_ear: float,
    ear_threshold: float,
    now: float,
    state: BlinkState,
    actions: MouseActions,
    config: DetectionConfig,
    control: DetectionControl,
) -> None:
    """Track blink start/end transitions based on EAR threshold crossings."""
    if smooth_ear < ear_threshold:
        if not state.in_blink:
            state.in_blink = True
            state.blink_start = now
        return

    if state.in_blink:
        _complete_blink_if_needed(now, state, actions, config, control)
        state.in_blink = False
        state.blink_start = None


def _dispatch_click_actions(
    now: float,
    state: BlinkState,
    actions: MouseActions,
    config: DetectionConfig,
) -> None:
    """Convert recorded blink timestamps into single/double blink actions."""
    if len(state.blink_times) >= 2:
        if (state.blink_times[-1] - state.blink_times[-2]) <= config.double_blink_max_gap_seconds:
            print("[EVENT] DOUBLE blink -> right click")
            actions.right_click()
            state.blink_times.clear()
    elif len(state.blink_times) == 1 and (now - state.blink_times[0]) > config.double_blink_max_gap_seconds:
        print("[EVENT] SINGLE blink -> left click")
        actions.left_click()
        state.blink_times.clear()


def _compute_fps(previous_timestamp: float | None, current_timestamp: float) -> tuple[float, float]:
    """Return current FPS and the updated frame timestamp."""
    if previous_timestamp is None:
        return 0.0, current_timestamp

    elapsed = current_timestamp - previous_timestamp
    if elapsed <= 0:
        return 0.0, current_timestamp

    return 1.0 / elapsed, current_timestamp


def _load_or_calibrate_threshold(
    cap: cv2.VideoCapture,
    face_mesh: Any,
    config: DetectionConfig,
    stop_check: Callable[[], bool] | None = None,
) -> tuple[float, bool]:
    """Use a saved threshold when available, otherwise calibrate and optionally save it."""
    saved_settings = load_runtime_settings() if config.use_saved_calibration else None
    if saved_settings is not None:
        same_camera = saved_settings.camera_index == config.camera_index
        same_sizes = saved_settings.process_size == config.process_size
        same_capture = saved_settings.camera_size == (config.camera_width, config.camera_height)
        if same_camera and same_sizes and same_capture:
            print("[INFO] Using saved calibration threshold.")
            return saved_settings.calibration_threshold, True

    threshold = calibrate_ear_threshold(
        cap,
        face_mesh,
        LEFT_EYE_LANDMARKS,
        RIGHT_EYE_LANDMARKS,
        config,
        stop_check=stop_check,
    )

    if config.save_calibration_after_run:
        save_runtime_settings(
            RuntimeSettings(
                calibration_threshold=threshold,
                camera_index=config.camera_index,
                process_size=config.process_size,
                camera_size=(config.camera_width, config.camera_height),
            )
        )

    return threshold, False


def run_detection(config: DetectionConfig | None = None, control: DetectionControl | None = None) -> None:
    """Start webcam and run blink detection until user presses ESC."""
    config = config or DetectionConfig()
    control = control or DetectionControl()
    control.mark_started()

    cap = _open_camera(config.camera_index)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam. Check camera permissions or another app using it.")
        control.mark_stopped()
        return

    pyautogui_actions = MouseActions()
    disabled_actions = NoOpMouseActions()

    try:
        model_path = ensure_model_available()
        base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
        landmarker_options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_faces=config.max_num_faces,
            min_face_detection_confidence=config.min_detection_confidence,
            min_face_presence_confidence=config.min_detection_confidence,
            min_tracking_confidence=config.min_detection_confidence,
        )

        _configure_camera(cap, config)
        with mp_vision.FaceLandmarker.create_from_options(landmarker_options) as face_mesh:
            try:
                ear_threshold, using_saved_calibration = _load_or_calibrate_threshold(
                    cap,
                    face_mesh,
                    config,
                    stop_check=control.should_stop,
                )
            except Exception as exc:
                print(f"[WARN] Calibration failed: {exc}")
                ear_threshold = config.fallback_ear_threshold
                using_saved_calibration = False

            ear_history: deque[float] = deque(maxlen=config.ear_smooth_window)
            state = BlinkState()
            previous_frame_timestamp: float | None = None

            # Pre-create drawing specs once to reduce per-frame allocations.
            mesh_spec = mp_vision.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
            contour_spec = mp_vision.drawing_utils.DrawingSpec(color=(0, 128, 255), thickness=1)

            print("[INFO] Detection started. Press ESC to quit.")

            while True:
                if control.should_stop():
                    print("[INFO] Stop requested.")
                    break

                ok, frame = _read_frame(cap)
                if not ok:
                    print("[WARN] Could not read frame from camera.")
                    break
                if frame is None:
                    print("[WARN] Camera returned an empty frame.")
                    break

                frame_small = cv2.resize(frame, config.process_size)
                display_frame = cv2.resize(frame_small, config.frame_size)
                rgb = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                timestamp_ms = time.monotonic_ns() // 1_000_000
                results = face_mesh.detect_for_video(mp_image, timestamp_ms)
                fps, previous_frame_timestamp = _compute_fps(previous_frame_timestamp, time.monotonic())
                active_threshold = control.get_threshold_override() or ear_threshold

                if results.face_landmarks:
                    if control.consume_recalibration_request():
                        print("[INFO] Recalibration requested.")
                        ear_threshold = calibrate_ear_threshold(
                            cap,
                            face_mesh,
                            LEFT_EYE_LANDMARKS,
                            RIGHT_EYE_LANDMARKS,
                            config,
                            stop_check=control.should_stop,
                        )
                        state = BlinkState()
                        ear_history.clear()

                    landmarks = results.face_landmarks[0]
                    left_ear = calculate_ear(LEFT_EYE_LANDMARKS, landmarks)
                    right_ear = calculate_ear(RIGHT_EYE_LANDMARKS, landmarks)
                    current_ear = (left_ear + right_ear) / 2.0
                    smooth_ear = _smooth_ear(ear_history, current_ear)
                    now = time.monotonic()
                    active_actions = (
                        pyautogui_actions
                        if control.is_cursor_control_enabled()
                        else disabled_actions
                    )

                    _update_blink_state(
                        smooth_ear,
                        active_threshold,
                        now,
                        state,
                        active_actions,
                        config,
                        control,
                    )
                    _dispatch_click_actions(now, state, active_actions, config)
                    control.update_live_stats(
                        fps=fps,
                        ear=smooth_ear,
                        threshold=active_threshold,
                    )

                    draw_status_overlay(
                        display_frame,
                        smooth_ear=smooth_ear,
                        ear_threshold=active_threshold,
                        fps=fps,
                        help_enabled=config.show_help_overlay,
                        using_saved_calibration=using_saved_calibration,
                    )
                    mp_vision.drawing_utils.draw_landmarks(
                        display_frame,
                        results.face_landmarks[0],
                        mp_vision.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION,
                        mesh_spec,
                        contour_spec,
                    )
                else:
                    draw_no_face_overlay(display_frame)
                    control.update_live_stats(
                        fps=fps,
                        ear=None,
                        threshold=active_threshold,
                    )

                cv2.imshow("Eye Blink Mouse Control", display_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    break
                if key in (ord("q"), ord("Q")):
                    print("[INFO] Quit requested by user.")
                    break
                if key in (ord("r"), ord("R")):
                    control.request_recalibration()
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    except cv2.error as exc:
        print(f"[ERROR] OpenCV failure: {exc}")
    except Exception as exc:
        print(f"[ERROR] Unexpected failure: {exc}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        control.mark_stopped()
