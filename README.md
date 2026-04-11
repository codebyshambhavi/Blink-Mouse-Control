# Blink Mouse Control

Hands-free mouse click automation powered by eye blinks and real-time facial landmark tracking.

## 🚀 Features

- Real-time face tracking with MediaPipe FaceMesh
- Blink detection using Eye Aspect Ratio (EAR)
- Single blink for left click
- Double blink for right click
- Long blink for click-and-hold behavior
- Automatic EAR calibration at startup
- Modular Python package structure suitable for scaling and maintenance
- CLI options for camera index and calibration duration

## 🛠️ Tech Stack

- Python
- OpenCV
- MediaPipe
- NumPy
- PyAutoGUI

## 📸 Demo

Add a GIF or short screen recording here.

Example:

![Demo](assets/demo.gif)

## ⚙️ Installation

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

## ▶️ Usage

Run the application:

```powershell
python src\main.py
```

Or run with CLI options:

```powershell
python src\main.py --camera-index 0 --calibration-seconds 4
```

On Windows, you can also use:

```powershell
.\run.bat
```

When the app starts, calibration runs for a few seconds. Keep your face centered and look at the camera until the webcam window opens.

Suggested demo workflow:

- Open Notepad, Paint, or any text field where clicks are easy to see.
- Keep good lighting and avoid moving too much during calibration.
- Use single blinks, double blinks, and long blinks to test the mouse actions.

## 📂 Project Structure

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
      └── ear.py
```

## 🤝 Contributing

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

## 📜 License

This project is released under the MIT License.

## 🙋‍♀️ Author

**Shambhavi**
