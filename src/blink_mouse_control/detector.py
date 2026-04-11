"""Real-time blink detection loop."""

import time
from collections import deque
from dataclasses import dataclass, field

import cv2
import mediapipe as mp

from .actions import MouseActions
from .calibration import calibrate_ear_threshold
from .config import DetectionConfig, LEFT_EYE_LANDMARKS, RIGHT_EYE_LANDMARKS
from .ear import calculate_ear


@dataclass
class BlinkState:
    """Mutable state used to infer blink patterns across frames."""

    in_blink: bool = False
    blink_start: float | None = None
    blink_times: list[float] = field(default_factory=list)


def _resolve_blink_actions(
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


def run_detection(config: DetectionConfig | None = None) -> None:
    """Start webcam and run blink detection until user presses ESC."""
    config = config or DetectionConfig()

    cap = cv2.VideoCapture(config.camera_index)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        return

    pyautogui_actions = MouseActions()
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils

    with mp_face_mesh.FaceMesh(
        max_num_faces=config.max_num_faces,
        refine_landmarks=config.refine_landmarks,
        min_detection_confidence=config.min_detection_confidence,
    ) as face_mesh:
        try:
            ear_threshold = calibrate_ear_threshold(
                cap,
                face_mesh,
                LEFT_EYE_LANDMARKS,
                RIGHT_EYE_LANDMARKS,
                config,
            )
        except Exception as exc:
            print(f"[WARN] Calibration failed: {exc}")
            ear_threshold = config.fallback_ear_threshold

        ear_history: deque[float] = deque(maxlen=config.ear_smooth_window)
        state = BlinkState()

        # Pre-create drawing specs once to reduce per-frame allocations.
        mesh_spec = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
        contour_spec = mp_drawing.DrawingSpec(color=(0, 128, 255), thickness=1)

        print("[INFO] Detection started. Press ESC to quit.")

        while True:
            ok, frame = cap.read()
            if not ok:
                print("[WARN] Could not read frame from camera.")
                break

            frame_small = cv2.resize(frame, config.frame_size)
            rgb = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                left_ear = calculate_ear(LEFT_EYE_LANDMARKS, landmarks)
                right_ear = calculate_ear(RIGHT_EYE_LANDMARKS, landmarks)
                current_ear = (left_ear + right_ear) / 2.0
                ear_history.append(current_ear)
                smooth_ear = sum(ear_history) / len(ear_history)
                now = time.monotonic()

                if smooth_ear < ear_threshold:
                    if not state.in_blink:
                        state.in_blink = True
                        state.blink_start = now
                elif state.in_blink:
                    if state.blink_start is not None:
                        duration = now - state.blink_start
                        state.blink_times.append(now)
                        if duration >= config.long_blink_min_duration_seconds:
                            print(f"[EVENT] LONG blink ({duration:.2f}s)")
                            pyautogui_actions.hold_left_click(config.click_hold_duration_seconds)
                            state.blink_times.clear()
                    state.in_blink = False
                    state.blink_start = None

                _resolve_blink_actions(now, state, pyautogui_actions, config)

                cv2.putText(
                    frame_small,
                    f"EAR: {smooth_ear:.3f}  Thr: {ear_threshold:.3f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (10, 255, 10),
                    2,
                )

                mp_drawing.draw_landmarks(
                    frame_small,
                    results.multi_face_landmarks[0],
                    mp_face_mesh.FACEMESH_TESSELATION,
                    mesh_spec,
                    contour_spec,
                )
            else:
                cv2.putText(
                    frame_small,
                    "Face not detected - align your face",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )

            cv2.imshow("Eye Blink Mouse Control", frame_small)
            if (cv2.waitKey(1) & 0xFF) == 27:
                break

    cap.release()
    cv2.destroyAllWindows()
