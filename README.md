# Deep Learning Based Air Gesture Text Recognition

An AI-powered real-time air writing and character recognition system using Deep Learning, Computer Vision, MediaPipe Hand Tracking, and Convolutional Neural Networks (CNN).

---

## Project Overview

Deep Learning Based Air Gesture Text Recognition is an intelligent Human-Computer Interaction system that allows users to write characters in the air using finger gestures without requiring physical input devices such as keyboards, touchscreens, or stylus pens.

The system captures real-time video through a webcam, tracks hand landmarks using MediaPipe, generates virtual air-writing strokes, and predicts characters using a trained CNN model.

This project combines:
- Artificial Intelligence
- Deep Learning
- Computer Vision
- Gesture Recognition
- Human-Computer Interaction (HCI)

to provide a touchless virtual writing experience.

---

# Key Features

- Real-time webcam-based air writing
- Hand and fingertip tracking using MediaPipe
- CNN-based handwritten character recognition
- Virtual air-writing canvas
- Real-time prediction display
- Live confidence visualization
- Touchless Human-Computer Interaction
- Voice output support
- FPS monitoring
- Modern AI-based graphical interface

---

# Technologies Used

| Technology | Purpose |
|------------|---------|
| Python | Core programming language |
| OpenCV | Webcam handling and image processing |
| MediaPipe | Hand tracking and landmark detection |
| TensorFlow | Deep Learning framework |
| Keras | CNN model implementation |
| NumPy | Numerical processing |
| CNN | Character recognition |
| Computer Vision | Gesture and image analysis |

---

# System Workflow

1. Webcam captures live video input.
2. MediaPipe detects hand landmarks.
3. Fingertip movement is tracked.
4. Finger trajectory is converted into stroke patterns.
5. Stroke image is preprocessed.
6. CNN model predicts the written character.
7. Output is displayed in real time.

---

# Project Architecture

```text
Webcam Input
      ↓
Hand Detection using MediaPipe
      ↓
Finger Motion Tracking
      ↓
Virtual Stroke Generation
      ↓
Image Preprocessing
      ↓
CNN Character Recognition
      ↓
Prediction Output Display

Modules:
1. Video Capture Module

Captures real-time webcam frames for processing.

2. Hand Detection Module

Detects hand landmarks and fingertip coordinates using MediaPipe.

3. Motion Tracking Module

Tracks finger movement trajectory during air writing.

4. Character Recognition Module

Recognizes air-written characters using a CNN model.

5. Output Display Module

Displays prediction results, confidence score, and virtual canvas.

Hardware Requirements:
Intel i5 Processor or higher
8GB RAM or above
HD Webcam
256GB Storage
Keyboard and Mouse
GPU (Optional for faster training)

Software Requirements:
Python 3.10
TensorFlow
OpenCV
MediaPipe
Keras
NumPy

Installation:
Step 1: Clone Repository
git clone https://github.com/your-username/Deep-Learning-Based-Air-Gesture-Text-Recognition.git
cd Deep-Learning-Based-Air-Gesture-Text-Recognition
Step 2: Create Virtual Environment
Windows
python -m venv venv
venv\Scripts\activate
Linux / Mac
python3 -m venv venv
source venv/bin/activate
Step 3: Install Dependencies
pip install -r requirements.txt
Run the Project
python main.py

The System displays:
Real-time hand tracking
Virtual writing canvas
Predicted character
Confidence score
Live prediction panel
FPS monitor

Limitations:
Sensitive to lighting conditions
Recognition depends on webcam quality
Fast hand movement may reduce accuracy
Background noise can affect tracking
Limited continuous sentence recognition

Future Enhancements:
Full sentence recognition
Multilingual support
Mobile application development
Advanced gesture controls
Improved AI prediction accuracy
Cloud-based recognition system
AR/VR integration

Applications:
Smart classrooms
Virtual keyboards
Gesture-controlled systems
Touchless interaction environments
Healthcare systems
Augmented Reality interfaces
Human-Computer Interaction systems

Algorithms Used:
MediaPipe Hand Tracking
Used for real-time hand landmark detection and fingertip tracking.
Convolutional Neural Network (CNN)
Used for image-based air-written character recognition.
OpenCV Image Processing
Used for webcam processing, stroke generation, and graphical rendering

## Project Preview
<p align="center">
  <img src="https://github.com/user-attachments/assets/bb9e2a04-b292-4e79-8140-abfbfd79414d" width="900">
</p>

