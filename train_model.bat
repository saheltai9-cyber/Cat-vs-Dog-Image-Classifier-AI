@echo off
setlocal enabledelayedexpansion

title PawVision AI — Model Training Pipeline

cd /d "%~dp0"
echo =====================================================================
echo          PAWVISION AI - RETRAIN MODEL PIPELINE
echo =====================================================================
echo.

if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
)

echo [*] Step 1: Validating Dataset...
python download_data.py
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Dataset preparation failed.
    pause
    exit /b 1
)

echo.
echo [*] Step 2: Training MobileNetV2 with Transfer Learning...
python train.py %*
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Training run failed.
    pause
    exit /b 1
)

echo.
echo =====================================================================
echo  [+] Model training and evaluation successfully completed!
echo  [+] Artifact saved: models\cat_dog_mobilenetv2.keras
echo  [+] Report updated: reports\model_metrics.json
echo =====================================================================
pause

