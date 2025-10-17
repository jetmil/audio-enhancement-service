@echo off
echo ================================================================================
echo Starting Audio Enhancement Service (Python 3.11 with AudioSR)
echo ================================================================================
echo.

if not exist "venv311\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found!
    echo Please run install_python311.bat first
    pause
    exit /b 1
)

echo Activating virtual environment...
call venv311\Scripts\activate.bat

echo.
echo Starting hybrid Gradio + FastAPI server with AudioSR...
echo.
echo Gradio Interface will be available at: http://localhost:7860
echo FastAPI Docs will be available at: http://localhost:7860/docs
echo.
echo Press Ctrl+C to stop the service
echo.

python audio_service.py
