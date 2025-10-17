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
echo Upgrading pip and build tools...
python -m pip install --upgrade pip setuptools wheel

echo.
echo Installing dependencies (this may take 5-10 minutes)...
pip install --prefer-binary -r requirements.txt

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
