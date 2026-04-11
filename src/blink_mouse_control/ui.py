"""Overlay helpers for user-facing feedback."""

from __future__ import annotations

from typing import Any

import cv2


def draw_status_overlay(
    frame: Any,
    *,
    smooth_ear: float | None,
    ear_threshold: float,
    fps: float,
    help_enabled: bool,
    using_saved_calibration: bool,
) -> None:
    """Draw status text that helps users understand what the app is doing."""
    lines = []
    if smooth_ear is not None:
        lines.append(f"EAR: {smooth_ear:.3f}  Thr: {ear_threshold:.3f}")
    else:
        lines.append(f"Thr: {ear_threshold:.3f}")

    lines.append(f"FPS: {fps:.1f}")
    lines.append("Saved calibration: on" if using_saved_calibration else "Saved calibration: off")

    y = 30
    for line in lines:
        cv2.putText(
            frame,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (10, 255, 10),
            2,
        )
        y += 26

    if help_enabled:
        cv2.putText(
            frame,
            "ESC/Q: quit | R: recalibrate | S: save threshold",
            (10, y + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 220, 120),
            2,
        )


def draw_no_face_overlay(frame: Any) -> None:
    """Display guidance when no face landmarks are detected."""
    cv2.putText(
        frame,
        "Face not detected - align your face",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2,
    )
