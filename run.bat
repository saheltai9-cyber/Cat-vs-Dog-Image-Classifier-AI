@echo off
setlocal enabledelayedexpansion

title PawVision AI — Cat vs Dog Classifier

:: 1. Move to Project Directory
cd /d "%~dp0"
echo =====================================================================
echo          PAWVISION AI - CAT VS DOG IMAGE CLASSIFIER
echo =====================================================================
echo.

:: 2. Check for Python
where python >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Python is not installed or not in PATH!
    echo Please install Python 3.11 or 3.12 (64-bit) from python.org.
    pause
    exit /b 1
)

:: 3. Create Local Virtual Environment if missing
if not exist ".venv\Scripts\activate.bat" (
    echo [*] Creating local virtual environment (.venv)...
    python -m venv .venv
    if %ERRORLEVEL% neq 0 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
)

:: 4. Activate Virtual Environment
call .venv\Scripts\activate.bat

:: 5. Install / Verify Dependencies
echo [*] Checking dependencies...
python -c "import tensorflow, nicegui, PIL, sklearn, plotly" >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [*] Installing required dependencies from requirements.txt...
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    if %ERRORLEVEL% neq 0 (
        echo [ERROR] Dependency installation failed.
        pause
        exit /b 1
    )
)

:: 6. Check Dataset & Download if missing
python -c "from download_data import is_dataset_ready; ready, _ = is_dataset_ready(); exit(0 if ready else 1)" >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [*] Dataset missing or incomplete. Downloading official dataset...
    python download_data.py
    if %ERRORLEVEL% neq 0 (
        echo [ERROR] Failed to download and extract dataset.
        pause
        exit /b 1
    )
)

:: 7. Check for Trained Model, Train if missing
if not exist "models\cat_dog_mobilenetv2.keras" (
    echo [*] No trained model found at models\cat_dog_mobilenetv2.keras.
    echo [*] Training MobileNetV2 transfer learning model now...
    python train.py
    if %ERRORLEVEL% neq 0 (
        echo [ERROR] Model training failed.
        pause
        exit /b 1
    )
)

:: 8. Launch Web Application
echo.
echo =====================================================================
echo  [+] PawVision AI is ready!
echo  [+] Launching Dashboard on: http://127.0.0.1:8080
echo =====================================================================
echo.
python app.py

pause

