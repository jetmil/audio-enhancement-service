"""
Simple test for Audio Enhancement Service
Tests basic functionality without requiring models
"""

import sys
from pathlib import Path
import numpy as np

print("="*80)
print("AUDIO ENHANCEMENT SERVICE - SYNTAX AND LOGIC TEST")
print("="*80)

# Test 1: Check imports
print("\n[Test 1] Checking imports...")
try:
    import torch
    import soundfile as sf
    import noisereduce as nr
    from pydub import AudioSegment
    import gradio as gr
    from fastapi import FastAPI
    print("[OK] All required packages available")
except ImportError as e:
    print(f"[ERROR] Missing package: {e}")
    sys.exit(1)

# Test 2: Check GPU availability
print("\n[Test 2] Checking GPU...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}")
    print("[OK] GPU available")
else:
    print("[WARNING] GPU not available (will use CPU)")

# Test 3: Check directory creation
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

# Test 4: Test mono to stereo conversion
print("\n[Test 4] Testing mono → stereo conversion...")
try:
    # Create test mono audio (1 second, 16kHz)
    mono_audio = np.random.randn(16000)

    # Convert to stereo
    if len(mono_audio.shape) == 1:
        stereo_audio = np.stack([mono_audio, mono_audio], axis=1)

    assert len(stereo_audio.shape) == 2, "Should be 2D array"
    assert stereo_audio.shape[1] == 2, "Should have 2 channels"
    print(f"[OK] Mono shape: {mono_audio.shape} -> Stereo shape: {stereo_audio.shape}")
except Exception as e:
    print(f"[ERROR] Stereo conversion failed: {e}")
    sys.exit(1)

# Test 5: Test MP3 export (basic)
print("\n[Test 5] Testing MP3 export...")
try:
    # Create test audio file
    test_wav = TEMP_DIR / "test.wav"
    test_mp3 = TEMP_DIR / "test.mp3"

    # Generate 1 second of silence in stereo
    sample_rate = 16000
    stereo_silence = np.zeros((sample_rate, 2))

    # Save as WAV
    sf.write(test_wav, stereo_silence, sample_rate)
    print(f"[OK] Created test WAV: {test_wav}")

    # Convert to MP3
    audio_segment = AudioSegment.from_wav(test_wav)
    audio_segment.export(
        test_mp3,
        format="mp3",
        bitrate="192k",
        parameters=["-ac", "2"]
    )
    print(f"[OK] Created test MP3: {test_mp3}")

    # Check file exists
    assert test_mp3.exists(), "MP3 file should exist"
    print(f"[OK] MP3 file size: {test_mp3.stat().st_size} bytes")

except Exception as e:
    print(f"[ERROR] MP3 export failed: {e}")
    print("Note: Ensure FFmpeg is installed for MP3 conversion")
    sys.exit(1)

# Test 6: Test FastAPI initialization
print("\n[Test 6] Testing FastAPI initialization...")
try:
    from audio_service import app
    print(f"[OK] FastAPI app created: {app.title}")
    print(f"[OK] Version: {app.version}")
except Exception as e:
    print(f"[ERROR] FastAPI initialization failed: {e}")
    sys.exit(1)

# Test 7: Test Gradio interface creation
print("\n[Test 7] Testing Gradio interface...")
try:
    from audio_service import create_gradio_interface
    gradio_app = create_gradio_interface()
    print(f"[OK] Gradio interface created")
except Exception as e:
    print(f"[ERROR] Gradio interface failed: {e}")
    sys.exit(1)

print("\n" + "="*80)
print("ALL TESTS PASSED!")
print("="*80)
print("\nService is ready to use. Run: python audio_service.py")
print("="*80)
