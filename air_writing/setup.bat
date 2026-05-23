@echo off
title AI Air Writing — Setup
color 0B
echo.
echo  ============================================================
echo    AI AIR WRITING RECOGNITION SYSTEM — Windows Setup
echo  ============================================================
echo.

python --version >nul 2>&1
if errorlevel 1 (
    echo  ERROR: Python not found.
    echo  Install from https://python.org  ^(tick "Add to PATH"^)
    pause & exit /b 1
)

echo  [1/5] Creating virtual environment...
python -m venv venv
if errorlevel 1 ( echo  FAILED & pause & exit /b 1 )

echo  [2/5] Activating venv...
call venv\Scripts\activate.bat

echo  [3/5] Upgrading pip...
python -m pip install --upgrade pip --quiet

echo  [4/5] Installing all packages...
pip install -r requirements.txt
if errorlevel 1 ( echo  Install FAILED & pause & exit /b 1 )

echo  [5/5] Creating folders...
mkdir datasets  2>nul
mkdir logs      2>nul
mkdir saved_text 2>nul
mkdir models\checkpoints 2>nul

echo.
echo  ============================================================
echo    SETUP COMPLETE!
echo.
echo    NEXT:
echo    1. Train the model (run once):
echo       python training\train.py
echo.
echo    2. Launch the app:
echo       python main.py
echo  ============================================================
echo.
pause
