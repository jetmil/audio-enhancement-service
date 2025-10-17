@echo off
echo ================================================================================
echo Audio Enhancement Service - Smart Installation with AudioSR
echo ================================================================================
echo.

echo Step 1: Checking Python versions...
python --version
echo.

echo Step 2: Creating main venv with current Python...
python -m venv venv
call venv\Scripts\activate.bat

echo.
echo Step 3: Installing core dependencies...
python -m pip install --upgrade pip setuptools wheel

echo.
echo Step 4: Installing Python 3.11-compatible packages...
pip install --prefer-binary -r requirements.txt

echo.
echo Step 5: Installing AudioSR workaround (optional)...
echo AudioSR requires Python 3.11 - checking if py -3.11 is available...
py -3.11 --version 2>nul
if errorlevel 1 (
    echo Python 3.11 not found. Service will work without AudioSR.
    echo To get AI enhancement, install Python 3.11 from python.org
    echo.
    echo Press any key to continue without AudioSR...
    pause >nul
) else (
    echo Python 3.11 found! Installing AudioSR in separate environment...
    py -3.11 -m pip install audiosr==0.0.7 numpy==1.23.5
    echo AudioSR installed successfully!
)

echo.
echo ================================================================================
echo Installation complete!
echo ================================================================================
echo.
if errorlevel 1 (
    echo Service installed without AudioSR AI enhancement
    echo Voice isolation and noise reduction will still work
) else (
    echo Service installed WITH AudioSR AI enhancement!
)
echo.
echo To start: start_service.bat
echo.
pause
