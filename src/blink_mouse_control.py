"""
blink_mouse_control.py
Core detection loop:
 - uses MediaPipe FaceMesh to get landmarks
 - computes EAR using utils.calculate_EAR
 - detects single / double / long blinks and triggers pyautogui actions
"""

import cv2
import mediapipe as mp
import pyautogui
import time
from collections import deque

# Import helpers from utils (same folder)
from utils import calculate_EAR, calibrate

# Eye landmark indices (MediaPipe FaceMesh)
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# Parameters (tweakable)
CALIBRATION_TIME = 4.0
EAR_SMOOTH_WINDOW = 6
DOUBLE_BLINK_MAX_GAP = 0.55
LONG_BLINK_MIN_DURATION = 0.45
CLICK_HOLD_DURATION = 0.5
FALLBACK_EAR_THRESHOLD = 0.22
CAMERA_INDEX = 0

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

def run_detection():
    """
    Start webcam, calibrate, then run blink detection loop.
    Actions:
      - SINGLE blink -> left click
      - DOUBLE blink -> right click
      - LONG blink -> hold left mouse briefly
    """
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return

    with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5) as face_mesh:
        # Calibration
        try:
            ear_threshold = calibrate(cap, face_mesh, LEFT_EYE, RIGHT_EYE, seconds=CALIBRATION_TIME)
        except Exception as e:
            print("[WARN] Calibration failed:", e)
            ear_threshold = FALLBACK_EAR_THRESHOLD

        ear_history = deque(maxlen=EAR_SMOOTH_WINDOW)
        blink_state = False
        blink_start = None
        blink_times = []

        print("[INFO] Detection started. Press ESC to quit.")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARN] Frame not read from camera.")
                break

            # resize for performance and consistent overlay
            frame_small = cv2.resize(frame, (960, 540))
            rgb = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)
            smooth_ear = None

            if results.multi_face_landmarks:
                lm = results.multi_face_landmarks[0].landmark
                left = calculate_EAR(LEFT_EYE, lm)
                right = calculate_EAR(RIGHT_EYE, lm)
                ear = (left + right) / 2.0
                ear_history.append(ear)
                smooth_ear = float(sum(ear_history) / len(ear_history))

                now = time.time()
                # Detect blink start / end
                if smooth_ear < ear_threshold:
                    if not blink_state:
                        blink_state = True
                        blink_start = now
                else:
                    if blink_state:
                        if blink_start is not None:
                            duration = now - blink_start
                            blink_times.append(now)
                            if duration >= LONG_BLINK_MIN_DURATION:
                                # LONG blink action
                                print(f"[EVENT] LONG blink ({duration:.2f}s)")
                                pyautogui.mouseDown()
                                time.sleep(CLICK_HOLD_DURATION)
                                pyautogui.mouseUp()
                                blink_times = []
                        # else: short blink; wait to see if double
                        blink_state = False
                        blink_start = None

                # Resolve single vs double
                if len(blink_times) >= 2:
                    if (blink_times[-1] - blink_times[-2]) <= DOUBLE_BLINK_MAX_GAP:
                        print("[EVENT] DOUBLE blink -> right click")
                        pyautogui.click(button='right')
                        blink_times = []
                elif len(blink_times) == 1:
                    # if older than gap, treat as single
                    if time.time() - blink_times[0] > DOUBLE_BLINK_MAX_GAP:
                        print("[EVENT] SINGLE blink -> left click")
                        pyautogui.click(button='left')
                        blink_times = []

                # Overlay information on frame
                if smooth_ear is not None:
                    cv2.putText(frame_small, f"EAR: {smooth_ear:.3f}  Thr: {ear_threshold:.3f}",
                                (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (10,255,10), 2)

                mp_drawing.draw_landmarks(frame_small, results.multi_face_landmarks[0],
                                          mp_face_mesh.FACEMESH_TESSELATION,
                                          mp_drawing.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1),
                                          mp_drawing.DrawingSpec(color=(0,128,255), thickness=1))
            else:
                cv2.putText(frame_small, "Face not detected - align your face", (10,30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

            cv2.imshow("Eye Blink Mouse Control", frame_small)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break

    cap.release()
    cv2.destroyAllWindows()
