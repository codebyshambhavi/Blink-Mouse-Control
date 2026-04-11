# Blink Mouse Control

Hands-free mouse control using real-time facial landmark tracking and blink detection.

This project is designed for accessibility use cases, technical demonstrations, and internship portfolios.

## Features

- Real-time facial landmark tracking with MediaPipe FaceMesh
- Blink-based mouse actions for left click, right click, and hold behavior
- Eye Aspect Ratio (EAR) detection for blink recognition
- Persistent calibration to avoid repeated setup on every run
- On-screen overlay showing EAR, FPS, and threshold values
- Optimized frame processing for smoother real-time performance
- Command-line controls for camera and calibration settings
- Modular Python package structure for maintainability

## Tech Stack

- Python
- OpenCV
- MediaPipe
- NumPy
- PyAutoGUI

## Why This Project Stands Out

- Solves a practical accessibility problem rather than serving as a basic demo.
- Combines computer vision, calibration, state management, and user feedback.
- Demonstrates performance tuning and persistence for a better user experience.
- Uses a clean project structure suitable for real-world development.

## Demo

This demo shows real-time blink-based mouse control with eye tracking and a live overlay.

- Blink triggers mouse clicks
- Overlay displays EAR, threshold, and FPS
- Calibration runs at startup for stable detection

Add a short GIF or screen recording here to demonstrate the interaction.

![Demo](assets/demo.gif)

## Installation

1. Clone the repository and open it in VS Code.
2. Create and activate a virtual environment:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. Install the dependencies:

   ```powershell
   pip install -r requirements.txt
   ```

4. If PowerShell blocks script execution, allow it for the current session:

   ```powershell
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
   ```

## Usage

Run the application:

```powershell
python src\main.py
```

Or run with CLI options:

```powershell
python src\main.py --camera-index 0 --calibration-seconds 4
```

For smoother performance on slower laptops, you can also reduce processing size:

```powershell
python src\main.py --process-width 640 --process-height 360
```

On Windows, you can also use:

```powershell
.\run.bat
```

When the app starts, calibration runs for a few seconds. Keep your face centered and look at the camera until the webcam window opens.

Useful controls during runtime:

- Press `ESC` or `Q` to quit
- Use `--no-saved-calibration` to force a fresh calibration on startup
- Use `--no-help-overlay` to hide the status and shortcut overlay

Suggested demo workflow:

- Open Notepad, Paint, or any text field where clicks are easy to observe.
- Keep good lighting and remain centered during calibration.
- Test single blinks, double blinks, and long blinks to verify the actions.

## Project Structure

```text
Blink_Project/
├── pyproject.toml
├── README.md
├── requirements.txt
├── run.bat
└── src/
   ├── main.py
   └── blink_mouse_control/
      ├── __init__.py
      ├── __main__.py
      ├── actions.py
      ├── calibration.py
      ├── cli.py
      ├── config.py
      ├── detector.py
      ├── settings.py
      ├── ui.py
      └── ear.py
```

## Future Improvements

- Eye-movement cursor control
- Sensitivity presets for different users
- Keyboard shortcuts for quick mode switching
- Test suite and CI pipeline for automated quality checks

## Contributing

Contributions are welcome. If you'd like to improve the project:

- Fork the repository
- Create a feature branch
- Make your changes
- Test the app locally
- Submit a pull request

Please keep changes focused and include clear descriptions of any behavioral updates.

Recommended contribution checklist:

- Format and lint code before opening a PR
- Add or update tests for behavior changes
- Update README for any user-facing changes

## License

This project is released under the MIT License.

## Author

**Shambhavi**
