"""CLI entrypoint for Blink Mouse Control."""

import argparse

from .config import DetectionConfig
from .detector import run_detection
from .ui import launch_control_panel


def build_parser() -> argparse.ArgumentParser:
    """Create command-line parser for runtime options."""
    parser = argparse.ArgumentParser(description="Control mouse clicks with eye blinks.")
    parser.add_argument("--camera-index", type=int, default=0, help="Webcam index (default: 0)")
    parser.add_argument(
        "--calibration-seconds",
        type=float,
        default=4.0,
        help="Calibration duration in seconds (default: 4.0)",
    )
    parser.add_argument(
        "--process-width",
        type=int,
        default=640,
        help="Frame width used for blink detection (default: 640)",
    )
    parser.add_argument(
        "--process-height",
        type=int,
        default=360,
        help="Frame height used for blink detection (default: 360)",
    )
    parser.add_argument(
        "--camera-width",
        type=int,
        default=640,
        help="Requested camera capture width (default: 640)",
    )
    parser.add_argument(
        "--camera-height",
        type=int,
        default=360,
        help="Requested camera capture height (default: 360)",
    )
    parser.add_argument(
        "--no-saved-calibration",
        action="store_true",
        help="Always calibrate on startup instead of reusing a saved threshold.",
    )
    parser.add_argument(
        "--no-help-overlay",
        action="store_true",
        help="Hide the on-screen usage tips and status legend.",
    )
    parser.add_argument(
        "--ui",
        action="store_true",
        help="Launch the CustomTkinter desktop control panel.",
    )
    return parser


def main() -> None:
    """Run the application from CLI options."""
    args = build_parser().parse_args()
    config = DetectionConfig(
        camera_index=args.camera_index,
        calibration_time_seconds=args.calibration_seconds,
        process_size=(args.process_width, args.process_height),
        camera_width=args.camera_width,
        camera_height=args.camera_height,
        use_saved_calibration=not args.no_saved_calibration,
        show_help_overlay=not args.no_help_overlay,
    )

    print("Blink Mouse Control - starting...")
    if args.ui:
        launch_control_panel(config=config)
    else:
        run_detection(config=config)
    print("Program ended.")


if __name__ == "__main__":
    main()
