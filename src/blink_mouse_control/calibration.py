"""Calibration routines for personalized EAR threshold selection."""

import time
from collections.abc import Sequence

import cv2

from .config import DetectionConfig
from .ear import calculate_ear, compute_threshold_from_samples


def calibrate_ear_threshold(
    cap: cv2.VideoCapture,
    face_mesh: object,
    left_eye_indices: Sequence[int],
    right_eye_indices: Sequence[int],
    config: DetectionConfig,
) -> float:
    """Collect EAR samples for a short time and derive a threshold."""
    print(f"[CALIBRATION] Look at the camera for about {int(config.calibration_time_seconds)}s...")

    start = time.monotonic()
    ear_samples: list[float] = []

    while time.monotonic() - start < config.calibration_time_seconds:
        ok, frame = cap.read()
        if not ok:
            continue

        frame_small = cv2.resize(frame, config.calibration_preview_size)
        rgb = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            left_ear = calculate_ear(left_eye_indices, landmarks)
            right_ear = calculate_ear(right_eye_indices, landmarks)
            ear_samples.append((left_ear + right_ear) / 2.0)

        elapsed = int(time.monotonic() - start)
        cv2.putText(
            frame_small,
            f"Calibrating... {elapsed}s/{int(config.calibration_time_seconds)}s",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (50, 200, 50),
            2,
        )
        cv2.imshow("Calibration", frame_small)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cv2.destroyWindow("Calibration")

    if len(ear_samples) < 6:
        print("[CALIBRATION] Not enough samples, using fallback threshold.")
        return config.fallback_ear_threshold

    threshold = compute_threshold_from_samples(ear_samples)
    print(f"[CALIBRATION] Computed EAR threshold: {threshold:.3f}")
    return threshold
