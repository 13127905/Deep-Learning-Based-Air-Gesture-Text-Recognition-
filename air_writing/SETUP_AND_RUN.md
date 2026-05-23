# AI Air Writing — Complete Setup & Run Guide

## STEP-BY-STEP (VS Code, Windows)

---

### STEP 1 — Prerequisites

Install these if you don't have them:

| Tool | Link |
|------|------|
| Python 3.10 (recommended) | https://python.org/downloads |
| VS Code | https://code.visualstudio.com |
| Python extension for VS Code | Search "Python" by Microsoft in Extensions |

> **IMPORTANT**: During Python install, tick **"Add Python to PATH"**

---

### STEP 2 — Open Project in VS Code

1. Unzip `air_writing_system.zip` anywhere (e.g. `C:\Projects\air_writing`)
2. Open VS Code
3. `File → Open Folder` → select the `air_writing` folder
4. You should see: `main.py`, `utils/`, `models/`, `training/`, `gui/`

---

### STEP 3 — Open Terminal in VS Code

Press **Ctrl + `** (backtick) or go to `View → Terminal`

The terminal opens at the project root automatically.

---

### STEP 4 — Create Virtual Environment

```bash
python -m venv venv
```

Then activate it:

```bash
# Windows (PowerShell)
venv\Scripts\Activate.ps1

# Windows (CMD)
venv\Scripts\activate.bat

# macOS / Linux
source venv/bin/activate
```

After activation you'll see **(venv)** at the start of the terminal.

> **If PowerShell says "execution policy" error:**
> Run this once: `Set-ExecutionPolicy RemoteSigned -Scope CurrentUser`

---

### STEP 5 — Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This installs: OpenCV, MediaPipe, TensorFlow, pyttsx3, numpy, scipy, matplotlib, scikit-learn.

Takes **3-10 minutes** depending on internet speed.

---

### STEP 6 — Select Python Interpreter in VS Code

1. Press **Ctrl + Shift + P**
2. Type: `Python: Select Interpreter`
3. Press Enter
4. Choose: `.\venv\Scripts\python.exe` (Windows) or `./venv/bin/python` (Mac/Linux)

The bottom-left status bar will show the venv Python version.

---

### STEP 7 — Train the CNN Model (Run Once)

```bash
python training/train.py
```

Or press **F5** → choose **"Train CNN"** from the run menu.

This:
- Downloads EMNIST dataset (~300 MB) automatically
- Trains 30 epochs
- Saves model to `models/air_writing_cnn.keras`
- Saves training curves to `logs/training_curves.png`

**Time:** ~20 minutes on CPU, ~3 minutes on GPU

> You only need to do this ONCE. After that, just run main.py.

---

### STEP 8 — Run the Application

```bash
python main.py
```

Or press **F5** → choose **"Run App"**

The webcam window opens. You're live!

---

## HOW TO WRITE CHARACTERS

1. Hold your hand in front of the camera
2. Raise your **index finger** — a cyan dot tracks the tip
3. **Move your finger** to draw a character in the air
4. **Pinch** (touch thumb to index tip) to lift the pen
5. The AI predicts your character → appears in the text bar
6. Draw the next character!

---

## KEYBOARD SHORTCUTS

| Key | Action |
|-----|--------|
| SPACE | Add space to text |
| ENTER | Speak text via voice |
| C | Clear canvas |
| S | Save text to file |
| V | Toggle voice on/off |
| BACKSPACE | Delete last character |
| H | Show/hide help overlay |
| D | Toggle debug mode |
| ESC or Q | Quit |

## GESTURE CONTROLS

| Gesture | Action |
|---------|--------|
| Index finger raised | Draw in air |
| Pinch (thumb + index) | Lift pen → triggers prediction |
| Closed fist (hold 1s) | Clear canvas |

---

## TROUBLESHOOTING

| Problem | Fix |
|---------|-----|
| Webcam won't open | Change `CAMERA_INDEX` in `utils/config.py` to 1 or 2 |
| Model not found | Run `python training/train.py` first |
| Low FPS | Lower `CAMERA_WIDTH`/`CAMERA_HEIGHT` in config.py |
| Hand not detected | Better lighting; lower `MIN_DETECTION_CONFIDENCE` to 0.60 |
| Wrong characters | Write larger and slower; increase `STROKE_THICKNESS` |
| No voice/crash | Run with `python main.py --no-voice` |
| PowerShell error | Run `Set-ExecutionPolicy RemoteSigned -Scope CurrentUser` |
| mediapipe error | Make sure Python is 3.9, 3.10, or 3.11 (not 3.12) |
| TF not found | Ensure venv is active before pip install |

---

## COMMAND REFERENCE

```bash
# Normal run
python main.py

# No voice output
python main.py --no-voice

# Different camera
python main.py --camera 1

# Debug verbose output
python main.py --debug

# Train with custom settings
python training/train.py --epochs 30 --batch 128
```
