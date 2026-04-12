"""Overlay helpers for user-facing feedback."""

from __future__ import annotations

from typing import Any

import cv2


HUD_GREEN = (80, 220, 120)
HUD_RED = (80, 80, 220)
HUD_BLUE = (235, 140, 60)
HUD_TEXT = (235, 235, 235)
HUD_MUTED = (190, 190, 190)


def _blend_rect(frame: Any, x1: int, y1: int, x2: int, y2: int, color: tuple[int, int, int], alpha: float) -> None:
    """Draw a semi-transparent rectangle with fast alpha blending."""
    h, w = frame.shape[:2]
    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h))
    if x2 <= x1 or y2 <= y1:
        return

    roi = frame[y1:y2, x1:x2]
    overlay = roi.copy()
    overlay[:] = color
    cv2.addWeighted(overlay, alpha, roi, 1.0 - alpha, 0.0, roi)


def draw_status_overlay(
    frame: Any,
    *,
    smooth_ear: float | None,
    ear_threshold: float,
    fps: float,
    help_enabled: bool,
    using_saved_calibration: bool,
    blink_detected: bool = False,
    running: bool = True,
) -> None:
    """Draw a minimal HUD with status, fps, ear, and lightweight feedback."""
    height, width = frame.shape[:2]

    # Top HUD bar
    bar_height = 58
    _blend_rect(frame, 8, 8, width - 8, 8 + bar_height, (20, 20, 20), alpha=0.45)

    status_text = "RUNNING" if running else "INACTIVE"
    status_color = HUD_GREEN if running else HUD_RED
    cv2.putText(
        frame,
        status_text,
        (22, 44),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.88,
        status_color,
        2,
        cv2.LINE_AA,
    )

    ear_text = "-" if smooth_ear is None else f"{smooth_ear:.3f}"
    metrics_text = f"FPS {fps:.1f}    EAR {ear_text}"
    text_size = cv2.getTextSize(metrics_text, cv2.FONT_HERSHEY_SIMPLEX, 0.72, 2)[0]
    cv2.putText(
        frame,
        metrics_text,
        (width - text_size[0] - 20, 44),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        HUD_BLUE,
        2,
        cv2.LINE_AA,
    )

    # Optional concise blink feedback
    if blink_detected:
        badge_text = "BLINK DETECTED"
        badge_size = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)[0]
        bx1 = width - badge_size[0] - 28
        by1 = 70
        bx2 = width - 14
        by2 = 102
        _blend_rect(frame, bx1, by1, bx2, by2, (30, 90, 30), alpha=0.55)
        cv2.putText(
            frame,
            badge_text,
            (bx1 + 8, by1 + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            HUD_GREEN,
            2,
            cv2.LINE_AA,
        )

    # Small bottom hint bar
    if help_enabled:
        hint_text = "Q/ESC Quit   |   R Recalibrate"
        text_w, text_h = cv2.getTextSize(hint_text, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)[0]
        y1 = height - 34
        y2 = height - 8
        _blend_rect(frame, 8, y1, width - 8, y2, (20, 20, 20), alpha=0.42)
        cv2.putText(
            frame,
            hint_text,
            (max(14, (width - text_w) // 2), y1 + text_h + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            HUD_MUTED,
            1,
            cv2.LINE_AA,
        )

    # Keep this subtle and compact in the HUD to avoid clutter.
    calibration_text = "CAL SAVED" if using_saved_calibration else "CAL TEMP"
    cv2.putText(
        frame,
        calibration_text,
        (22, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        HUD_MUTED,
        1,
        cv2.LINE_AA,
    )

    threshold_text = f"THR {ear_threshold:.3f}"
    threshold_size = cv2.getTextSize(threshold_text, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 1)[0]
    cv2.putText(
        frame,
        threshold_text,
        (width - threshold_size[0] - 20, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        HUD_MUTED,
        1,
        cv2.LINE_AA,
    )


def draw_face_guides(
    frame: Any,
    landmarks: Any,
    *,
    left_eye_landmarks: tuple[int, ...],
    right_eye_landmarks: tuple[int, ...],
) -> None:
    """Draw only lightweight guides: face box and eye landmark points."""
    height, width = frame.shape[:2]

    xs: list[int] = []
    ys: list[int] = []
    for lm in landmarks:
        xs.append(int(lm.x * width))
        ys.append(int(lm.y * height))

    if not xs or not ys:
        return

    min_x = max(0, min(xs) - 10)
    min_y = max(0, min(ys) - 10)
    max_x = min(width - 1, max(xs) + 10)
    max_y = min(height - 1, max(ys) + 10)
    cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), HUD_BLUE, 1, cv2.LINE_AA)

    for index in left_eye_landmarks:
        lm = landmarks[index]
        cv2.circle(frame, (int(lm.x * width), int(lm.y * height)), 2, HUD_GREEN, -1, cv2.LINE_AA)
    for index in right_eye_landmarks:
        lm = landmarks[index]
        cv2.circle(frame, (int(lm.x * width), int(lm.y * height)), 2, HUD_GREEN, -1, cv2.LINE_AA)


def draw_no_face_overlay(frame: Any) -> None:
    """Display guidance when no face landmarks are detected."""
    draw_status_overlay(
        frame,
        smooth_ear=None,
        ear_threshold=0.0,
        fps=0.0,
        help_enabled=True,
        using_saved_calibration=False,
        blink_detected=False,
        running=False,
    )
    cv2.putText(
        frame,
        "Face not detected",
        (22, 92),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        HUD_RED,
        2,
        cv2.LINE_AA,
    )
