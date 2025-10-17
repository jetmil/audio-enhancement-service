@echo off
echo ================================================================================
echo Starting Audio Enhancement Service
echo ================================================================================
echo.

if not exist "venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found!
    echo Please run install.bat first
    pause
    exit /b 1
)

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo Starting hybrid Gradio + FastAPI server...
echo.
echo Gradio Interface will be available at: http://localhost:7861
echo FastAPI Docs will be available at: http://localhost:7861/docs
echo.
echo Press Ctrl+C to stop the service
echo.

python audio_service.py
