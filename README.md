# Blink Mouse Control

Mouse control with eye movements and blinking using computer vision.

## 🚀 Features

- Real-time face tracking with MediaPipe FaceMesh
- Blink detection using Eye Aspect Ratio (EAR)
- Single blink for left click
- Double blink for right click
- Long blink for click-and-hold behavior
- Automatic EAR calibration at startup
- Simple Windows launcher script for quick setup

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

1. Clone or open the project in VS Code.
2. Create and activate a virtual environment:

   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

3. Install the dependencies:

   ```powershell
   pip install -r requirements.txt
   ```

4. If PowerShell blocks script execution, run:

   ```powershell
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
   ```

## ▶️ Usage

Run the application with either of the following:

```powershell
python src\main.py
```

or on Windows:

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
├── README.md
├── requirements.txt
├── run.bat
├── presentation_notes.txt
└── src/
    ├── main.py
    ├── blink_mouse_control.py
    └── utils.py
```

## 🤝 Contributing

Contributions are welcome. If you'd like to improve the project:

- Fork the repository
- Create a feature branch
- Make your changes
- Test the app locally
- Submit a pull request

Please keep changes focused and include clear descriptions of any behavioral updates.

## 📜 License

License information has not been added yet. If you plan to publish this project, add a license file such as MIT, Apache 2.0, or GPL and update this section accordingly.

## 🙋‍♀️ Author

**Shambhavi**
