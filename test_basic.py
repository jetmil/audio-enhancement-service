"""
Basic syntax and structure test
Only tests core functionality without requiring all packages
"""

import sys
from pathlib import Path

print("="*80)
print("AUDIO ENHANCEMENT SERVICE - BASIC SYNTAX TEST")
print("="*80)

# Test 1: Check PyTorch
print("\n[Test 1] Checking PyTorch...")
try:
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA: {torch.version.cuda}")
    print("[OK] PyTorch available")
except ImportError as e:
    print(f"[ERROR] PyTorch not found: {e}")
    sys.exit(1)

# Test 2: Check Python syntax of main file
print("\n[Test 2] Checking Python syntax of audio_service.py...")
try:
    import py_compile
    py_compile.compile("audio_service.py", doraise=True)
    print("[OK] No syntax errors in audio_service.py")
except py_compile.PyCompileError as e:
    print(f"[ERROR] Syntax error: {e}")
    sys.exit(1)

# Test 3: Directory structure
print("\n[Test 3] Checking directory creation...")
try:
    OUTPUT_DIR = Path("outputs")
    OUTPUT_DIR.mkdir(exist_ok=True)
    TEMP_DIR = Path("temp")
    TEMP_DIR.mkdir(exist_ok=True)
    print(f"[OK] Created: {OUTPUT_DIR}")
    print(f"[OK] Created: {TEMP_DIR}")
except Exception as e:
    print(f"[ERROR] Directory creation failed: {e}")
    sys.exit(1)

# Test 4: File structure check
print("\n[Test 4] Checking file structure...")
required_files = [
    "audio_service.py",
    "requirements.txt",
    "install.bat",
    "start_service.bat",
    "README.md",
    ".gitignore"
]

missing = []
for file in required_files:
    if not Path(file).exists():
        missing.append(file)

if missing:
    print(f"[WARNING] Missing files: {missing}")
else:
    print("[OK] All required files present")

print("\n" + "="*80)
print("BASIC TESTS PASSED!")
print("="*80)
print("\nTo install dependencies: install.bat")
print("To start service: start_service.bat")
print("="*80)
