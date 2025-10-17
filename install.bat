@echo off
echo ================================================================================
echo Audio Enhancement Service - Installation
echo ================================================================================
echo.

echo Creating virtual environment...
python -m venv venv
if errorlevel 1 (
    echo ERROR: Failed to create virtual environment
    pause
    exit /b 1
)

echo.
echo Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo Installing dependencies...
python -m pip install --upgrade pip
pip install -r requirements.txt

if errorlevel 1 (
    echo.
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)

echo.
echo ================================================================================
echo Installation complete!
echo ================================================================================
echo.
echo To start the service, run: start_service.bat
echo.
pause
