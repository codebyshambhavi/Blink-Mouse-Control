"""Real-time blink detection orchestration."""

import os
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import cv2
import mediapipe as mp

from .actions import MouseActions
from .calibration import calibrate_ear_threshold
from .config import DetectionConfig, LEFT_EYE_LANDMARKS, RIGHT_EYE_LANDMARKS
from .ear import calculate_ear
from .settings import RuntimeSettings, load_runtime_settings, save_runtime_settings
from .ui import draw_no_face_overlay, draw_status_overlay


@dataclass
class BlinkState:
    """Mutable state used to infer blink patterns across frames."""

    in_blink: bool = False
    blink_start: float | None = None
    blink_times: list[float] = field(default_factory=list)


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
) -> None:
    """Finalize a blink and dispatch long-blink action when threshold is met."""
    if state.blink_start is None:
        return

    duration = now - state.blink_start
    state.blink_times.append(now)
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
) -> None:
    """Track blink start/end transitions based on EAR threshold crossings."""
    if smooth_ear < ear_threshold:
        if not state.in_blink:
            state.in_blink = True
            state.blink_start = now
        return

    if state.in_blink:
        _complete_blink_if_needed(now, state, actions, config)
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


def run_detection(config: DetectionConfig | None = None) -> None:
    """Start webcam and run blink detection until user presses ESC."""
    config = config or DetectionConfig()

    cap = _open_camera(config.camera_index)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam. Check camera permissions or another app using it.")
        return

    pyautogui_actions = MouseActions()
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils

    try:
        _configure_camera(cap, config)
        with mp_face_mesh.FaceMesh(
            max_num_faces=config.max_num_faces,
            refine_landmarks=config.refine_landmarks,
            min_detection_confidence=config.min_detection_confidence,
        ) as face_mesh:
            try:
                ear_threshold, using_saved_calibration = _load_or_calibrate_threshold(
                    cap,
                    face_mesh,
                    config,
                )
            except Exception as exc:
                print(f"[WARN] Calibration failed: {exc}")
                ear_threshold = config.fallback_ear_threshold
                using_saved_calibration = False

            ear_history: deque[float] = deque(maxlen=config.ear_smooth_window)
            state = BlinkState()
            previous_frame_timestamp: float | None = None

            # Pre-create drawing specs once to reduce per-frame allocations.
            mesh_spec = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
            contour_spec = mp_drawing.DrawingSpec(color=(0, 128, 255), thickness=1)

            print("[INFO] Detection started. Press ESC to quit.")

            while True:
                ok, frame = _read_frame(cap)
                if not ok:
                    print("[WARN] Could not read frame from camera.")
                    break

                frame_small = cv2.resize(frame, config.process_size)
                display_frame = cv2.resize(frame_small, config.frame_size)
                rgb = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb)
                fps, previous_frame_timestamp = _compute_fps(previous_frame_timestamp, time.monotonic())

                if results.multi_face_landmarks:
                    landmarks = results.multi_face_landmarks[0].landmark
                    left_ear = calculate_ear(LEFT_EYE_LANDMARKS, landmarks)
                    right_ear = calculate_ear(RIGHT_EYE_LANDMARKS, landmarks)
                    current_ear = (left_ear + right_ear) / 2.0
                    smooth_ear = _smooth_ear(ear_history, current_ear)
                    now = time.monotonic()

                    _update_blink_state(
                        smooth_ear,
                        ear_threshold,
                        now,
                        state,
                        pyautogui_actions,
                        config,
                    )
                    _dispatch_click_actions(now, state, pyautogui_actions, config)

                    draw_status_overlay(
                        display_frame,
                        smooth_ear=smooth_ear,
                        ear_threshold=ear_threshold,
                        fps=fps,
                        help_enabled=config.show_help_overlay,
                        using_saved_calibration=using_saved_calibration,
                    )
                    mp_drawing.draw_landmarks(
                        display_frame,
                        results.multi_face_landmarks[0],
                        mp_face_mesh.FACEMESH_TESSELATION,
                        mesh_spec,
                        contour_spec,
                    )
                else:
                    draw_no_face_overlay(display_frame)

                cv2.imshow("Eye Blink Mouse Control", display_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    break
                if key in (ord("q"), ord("Q")):
                    print("[INFO] Quit requested by user.")
                    break
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    except cv2.error as exc:
        print(f"[ERROR] OpenCV failure: {exc}")
    except Exception as exc:
        print(f"[ERROR] Unexpected failure: {exc}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
