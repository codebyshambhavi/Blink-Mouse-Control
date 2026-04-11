"""
utils.py
Helper functions for blink detection:
- EAR calculation
- simple calibration routine
"""

import numpy as np
import time

def euclidean(a, b):
    """Euclidean distance between two 2D points."""
    return np.linalg.norm(np.array(a) - np.array(b))

def calculate_EAR(eye_points, landmarks):
    """
    Calculate Eye Aspect Ratio (EAR) for one eye.
    - eye_points : list of 6 indices corresponding to MediaPipe FaceMesh landmarks for that eye
    - landmarks   : list of landmarks (each with .x and .y normalized coords)
    Returns scalar EAR.
    """
    coords = [(landmarks[i].x, landmarks[i].y) for i in eye_points]
    # vertical distances
    A = euclidean(coords[1], coords[5])
    B = euclidean(coords[2], coords[4])
    # horizontal distance
    C = euclidean(coords[0], coords[3])
    if C == 0:
        return 0.0
    ear = (A + B) / (2.0 * C)
    return ear

def compute_threshold_from_samples(ears):
    """
    Given a list/array of EAR samples (open-eye mostly), compute a heuristic threshold.
    Uses median and lower percentile heuristics to find a mid-point threshold.
    """
    ears = np.array(ears)
    if len(ears) == 0:
        return 0.22  # safe fallback
    open_median = np.median(ears)
    closed_10pct = np.percentile(ears, 10)
    thr1 = open_median * 0.7
    thr2 = closed_10pct * 1.3
    threshold = float(np.mean([thr1, thr2]))
    threshold = max(0.12, min(threshold, 0.35))
    return threshold

def calibrate(cap, face_mesh, left_eye_inds, right_eye_inds, seconds=4.0, resize=(640,360)):
    """
    Simple auto-calibration routine:
    - collects EAR samples for `seconds`
    - returns computed threshold
    """
    import cv2
    print(f"[CALIBRATION] Look at the camera for ~{int(seconds)}s...")

    start = time.time()
    ears = []
    while time.time() - start < seconds:
        ret, frame = cap.read()
        if not ret:
            continue
        frame_small = cv2.resize(frame, resize)
        rgb = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark
            left = calculate_EAR(left_eye_inds, lm)
            right = calculate_EAR(right_eye_inds, lm)
            ears.append((left + right) / 2.0)
        # show a simple preview during calibration
        cv2.putText(frame_small, f"Calibrating... {int(time.time()-start)}s/{int(seconds)}s",
                    (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50,200,50), 2)
        cv2.imshow("Calibration", frame_small)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cv2.destroyWindow("Calibration")
    if len(ears) < 6:
        print("[CALIBRATION] Not enough samples, using fallback 0.22")
        return 0.22
    threshold = compute_threshold_from_samples(ears)
    print(f"[CALIBRATION] Computed EAR threshold: {threshold:.3f}")
    return threshold
