"""Overlay helpers for user-facing feedback."""

from __future__ import annotations

from typing import Any

import cv2


HUD_GREEN = (80, 220, 120)
HUD_RED = (80, 80, 220)
HUD_BLUE = (235, 150, 70)
HUD_ORANGE = (60, 165, 255)
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
    blink_strength: float = 0.0,
    running: bool = True,
) -> None:
    """Draw a minimal HUD with status, fps, ear, and lightweight feedback."""
    height, width = frame.shape[:2]

    # Top HUD bar
    bar_height = 72
    _blend_rect(frame, 8, 8, width - 8, 8 + bar_height, (18, 18, 18), alpha=0.52)

    status_text = "RUNNING" if running else "STOPPED"
    status_color = HUD_GREEN if running else HUD_RED
    cv2.putText(
        frame,
        status_text,
        (24, 56),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        status_color,
        3,
        cv2.LINE_AA,
    )

    calibration_text = "CAL SAVED" if using_saved_calibration else "CAL TEMP"
    cv2.putText(
        frame,
        calibration_text,
        (24, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        HUD_MUTED,
        1,
        cv2.LINE_AA,
    )

    ear_text = "-" if smooth_ear is None else f"{smooth_ear:.3f}"
    fps_text = f"FPS {fps:.1f}"
    ear_text_block = f"EAR {ear_text}"
    threshold_text = f"THR {ear_threshold:.3f}"
    separator = "  |  "

    scale = 0.68
    thickness = 2
    fps_size = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0]
    ear_size = cv2.getTextSize(ear_text_block, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0]
    thr_size = cv2.getTextSize(threshold_text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0]
    sep_size = cv2.getTextSize(separator, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0]
    total_width = fps_size[0] + sep_size[0] + ear_size[0] + sep_size[0] + thr_size[0]
    x = width - total_width - 24
    y = 52

    cv2.putText(
        frame,
        fps_text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        HUD_BLUE,
        thickness,
        cv2.LINE_AA,
    )
    x += fps_size[0]
    cv2.putText(frame, separator, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, HUD_MUTED, 1, cv2.LINE_AA)
    x += sep_size[0]
    cv2.putText(frame, ear_text_block, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, HUD_BLUE, thickness, cv2.LINE_AA)
    x += ear_size[0]
    cv2.putText(frame, separator, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, HUD_MUTED, 1, cv2.LINE_AA)
    x += sep_size[0]
    cv2.putText(frame, threshold_text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, HUD_ORANGE, thickness, cv2.LINE_AA)

    # Blink indicator with smooth fade-out.
    if blink_strength > 0.0:
        intensity = max(0.0, min(blink_strength, 1.0))
        badge_text = "BLINK"
        badge_size = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.72, 2)[0]
        bx1 = width - badge_size[0] - 40
        by1 = 90
        bx2 = width - 16
        by2 = 126
        _blend_rect(frame, bx1, by1, bx2, by2, (26, 70, 26), alpha=0.20 + 0.35 * intensity)
        blink_text_color = (
            int(120 + 100 * intensity),
            int(180 + 60 * intensity),
            int(120 + 100 * intensity),
        )
        cv2.putText(
            frame,
            badge_text,
            (bx1 + 12, by1 + 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            blink_text_color,
            2,
            cv2.LINE_AA,
        )

    # Small bottom hint bar
    if help_enabled:
        hint_text = "Q or ESC: Quit    |    R: Recalibrate    |    B: Cycle Beauty"
        text_w, text_h = cv2.getTextSize(hint_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        y1 = height - 40
        y2 = height - 8
        _blend_rect(frame, 8, y1, width - 8, y2, (18, 18, 18), alpha=0.48)
        cv2.putText(
            frame,
            hint_text,
            (max(16, (width - text_w) // 2), y1 + text_h + 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
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
    cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (180, 122, 70), 1, cv2.LINE_AA)

    for index in left_eye_landmarks:
        lm = landmarks[index]
        cv2.circle(frame, (int(lm.x * width), int(lm.y * height)), 3, (135, 225, 165), -1, cv2.LINE_AA)
    for index in right_eye_landmarks:
        lm = landmarks[index]
        cv2.circle(frame, (int(lm.x * width), int(lm.y * height)), 3, (135, 225, 165), -1, cv2.LINE_AA)


def draw_no_face_overlay(
    frame: Any,
    *,
    fps: float,
    ear_threshold: float,
    help_enabled: bool,
    using_saved_calibration: bool,
) -> None:
    """Display guidance when no face landmarks are detected."""
    draw_status_overlay(
        frame,
        smooth_ear=None,
        ear_threshold=ear_threshold,
        fps=fps,
        help_enabled=help_enabled,
        using_saved_calibration=using_saved_calibration,
        blink_strength=0.0,
        running=True,
    )
    cv2.putText(
        frame,
        "Face not detected",
        (24, 112),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.74,
        HUD_RED,
        2,
        cv2.LINE_AA,
    )
