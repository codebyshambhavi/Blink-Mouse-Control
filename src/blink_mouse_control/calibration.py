"""Calibration routines for personalized EAR threshold selection."""

import time
from collections.abc import Sequence
from typing import Any

import cv2
import mediapipe as mp

from .config import DetectionConfig
from .ear import calculate_ear, compute_threshold_from_samples


def _extract_average_ear(
    results: Any,
    left_eye_indices: Sequence[int],
    right_eye_indices: Sequence[int],
) -> float | None:
    """Return average EAR for detected face landmarks, else None."""
    if not results.face_landmarks:
        return None

    landmarks = results.face_landmarks[0]
    left_ear = calculate_ear(left_eye_indices, landmarks)
    right_ear = calculate_ear(right_eye_indices, landmarks)
    return (left_ear + right_ear) / 2.0


def _draw_calibration_overlay(frame: Any, elapsed_seconds: int, total_seconds: float) -> None:
    """Render calibration progress text on the preview frame."""
    cv2.putText(
        frame,
        f"Calibrating... {elapsed_seconds}s/{int(total_seconds)}s",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (50, 200, 50),
        2,
    )


def calibrate_ear_threshold(
    cap: cv2.VideoCapture,
    face_mesh: object,
    left_eye_indices: Sequence[int],
    right_eye_indices: Sequence[int],
    config: DetectionConfig,
) -> float:
    """Collect EAR samples for a short time and derive a threshold."""
    print(
        f"[CALIBRATION] Look at the camera for about {int(config.calibration_time_seconds)}s..."
    )

    start = time.monotonic()
    ear_samples: list[float] = []

    while time.monotonic() - start < config.calibration_time_seconds:
        try:
            ok, frame = cap.read()
        except cv2.error as exc:
            print(f"[CALIBRATION] Camera error: {exc}")
            break

        if not ok:
            continue

        frame_small = cv2.resize(frame, config.calibration_preview_size)
        rgb = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = time.monotonic_ns() // 1_000_000
        results = face_mesh.detect_for_video(mp_image, timestamp_ms)

        average_ear = _extract_average_ear(results, left_eye_indices, right_eye_indices)
        if average_ear is not None:
            ear_samples.append(average_ear)

        elapsed = int(time.monotonic() - start)
        _draw_calibration_overlay(frame_small, elapsed, config.calibration_time_seconds)
        cv2.imshow("Calibration", frame_small)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            print("[CALIBRATION] Cancelled by user.")
            break
        if key in (ord("q"), ord("Q")):
            print("[CALIBRATION] Quit requested by user.")
            break

    cv2.destroyWindow("Calibration")

    if len(ear_samples) < 6:
        print("[CALIBRATION] Not enough samples, using fallback threshold.")
        return config.fallback_ear_threshold

    threshold = compute_threshold_from_samples(ear_samples)
    print(f"[CALIBRATION] Computed EAR threshold: {threshold:.3f}")
    return threshold
