@echo off
echo ================================================================================
echo Audio Enhancement Service - Installation for Python 3.11 (with AudioSR)
echo ================================================================================
echo.
echo IMPORTANT: This requires Python 3.11, not 3.13!
echo AudioSR does not support Python 3.13 yet.
echo.
echo If you have Python 3.13, use install.bat instead (without AudioSR).
echo.
pause

echo Creating virtual environment with Python 3.11...
py -3.11 -m venv venv311
if errorlevel 1 (
    echo ERROR: Python 3.11 not found!
    echo Please install Python 3.11 from https://www.python.org/downloads/
    pause
    exit /b 1
)

echo.
echo Activating virtual environment...
call venv311\Scripts\activate.bat

echo.
echo Upgrading pip and build tools...
python -m pip install --upgrade pip setuptools wheel

echo.
echo Installing dependencies with AudioSR (this may take 10-15 minutes)...
pip install --prefer-binary -r requirements_audiosr.txt

if errorlevel 1 (
    echo.
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)

echo.
echo ================================================================================
echo Installation complete with AudioSR!
echo ================================================================================
echo.
echo To start the service, run: start_service_python311.bat
echo.
pause
