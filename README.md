# AI-Based Eye Blink Detection (Blink Mouse Control)

## Summary
Real-time blink detection using MediaPipe FaceMesh and Eye Aspect Ratio (EAR).
Actions:
- Single blink -> Left click
- Double blink -> Right click
- Long blink -> Hold left click briefly

## Setup (Windows)
1. Open project folder in VS Code.
2. Activate venv in terminal:
   .\venv\Scripts\Activate.ps1
   (If PowerShell blocks: Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process)
3. Install dependencies (if not already):
   pip install -r requirements.txt
4. Run:
   python src\main.py
   or double-click run.bat

## Demo tips
- Open Notepad or Paint and place the cursor where clicks will be visible.
- Use consistent lighting and keep your face centered in the webcam during calibration.
- Calibration will run for ~4 seconds when the program starts.

## Notes
- Tested on Python 3.10 with packages: mediapipe, opencv-python, numpy, pyautogui.
- On macOS you must allow Accessibility permission for synthetic mouse actions.
