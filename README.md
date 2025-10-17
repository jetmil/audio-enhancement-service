# 🎵 Audio Enhancement Service

AI-Powered Audio Enhancement with Voice Isolation and Quality Upscaling

**Optimized for RTX 3090 | CUDA 12.4**

## 🎯 Features

### 3-in-1 Audio Enhancement Pipeline

1. **Voice Isolation (Demucs v4)** - Separate vocals from background music/noise
2. **Noise Reduction (Neural)** - Remove static, hum, and ambient noise
3. **Audio Super-Resolution (AudioSR)** - Enhance quality and upscale to 48kHz

### Output Format
- **MP3 Stereo 192 kbps** (converts mono input to stereo)
- Professional broadcast quality
- Compatible with all media players

### Dual Interface

- **Gradio Web UI** - Interactive drag-and-drop interface
- **FastAPI REST API** - Programmatic access for automation

## 🚀 Quick Start

### 1. Installation

```batch
install.bat
```

This will:
- Create Python virtual environment
- Install all dependencies (AudioSR, Demucs, Gradio, FastAPI, etc.)
- Download required AI models

### 2. Start Service

```batch
start_service.bat
```

The service will launch on **http://localhost:7860**

## 📖 Usage

### Web Interface (Gradio)

1. Open http://localhost:7860 in your browser
2. Upload an audio file (WAV, MP3, FLAC, etc.)
3. Select enhancement options:
   - ✅ Voice Isolation - removes background music
   - ✅ Noise Reduction - removes static/hum
   - ✅ Super-Resolution - enhances quality to 48kHz
4. Choose target sample rate (16kHz - 48kHz)
5. Click "🚀 Enhance Audio"
6. Download enhanced MP3 stereo 192 kbps

### API Interface (FastAPI)

#### API Documentation
Visit http://localhost:7860/docs for interactive API docs

#### Example: Enhance Audio via API

```python
import requests

# Upload and enhance
with open("input.wav", "rb") as f:
    response = requests.post(
        "http://localhost:7860/enhance",
        files={"file": f},
        params={
            "enable_demucs": True,
            "enable_noise_reduction": True,
            "enable_sr": True,
            "target_sr": 48000
        }
    )

result = response.json()
print(f"Enhanced file: {result['output_file']}")
print(f"Download: http://localhost:7860{result['download_url']}")
```

#### Example: Download Enhanced File

```python
import requests

# Download result
response = requests.get("http://localhost:7860/outputs/enhanced_audio.mp3")
with open("enhanced_audio.mp3", "wb") as f:
    f.write(response.content)
```

## 🎛️ Enhancement Options

### Voice Isolation (Demucs v4)

**Best for**: Removing background music from speech, isolating vocals

- Uses Hybrid Transformer architecture
- Separates vocals from accompaniment
- Optimized for RTX 3090 with CUDA acceleration

**When to use**:
- ✅ Speech with background music
- ✅ Podcast with intro/outro music
- ✅ Karaoke recordings
- ❌ Already clean speech (unnecessary overhead)

### Noise Reduction (Neural)

**Best for**: Removing static, hum, ambient noise

- Neural network-based spectral gating
- Removes stationary noise (AC hum, white noise, etc.)
- Preserves voice quality

**When to use**:
- ✅ Recordings with background hum/static
- ✅ Outdoor recordings with wind noise
- ✅ Low-quality microphone recordings
- ❌ Already clean studio recordings

### Audio Super-Resolution (AudioSR)

**Best for**: Enhancing quality and upscaling sample rate

- Diffusion-based AI model
- Upscales any sample rate to 48kHz
- Enhances clarity and richness

**When to use**:
- ✅ Low-quality recordings (8kHz, 16kHz)
- ✅ Telephone/voice call recordings
- ✅ Old/degraded audio files
- ✅ Final quality boost for all audio

## 🔧 Technical Details

### System Requirements

- **GPU**: NVIDIA RTX 3090 (or any CUDA-compatible GPU)
- **CUDA**: 12.4+ (already installed with PyTorch)
- **RAM**: 16GB+ recommended
- **Python**: 3.9 - 3.11

### Dependencies

- **PyTorch 2.6.0** (CUDA 12.4) - already installed
- **AudioSR 0.0.7** - diffusion-based super-resolution
- **Demucs 4.0.1** - Hybrid Transformer source separation
- **noisereduce 3.0.2** - neural noise suppression
- **Gradio 4.44.0** - web UI framework
- **FastAPI 0.115.0** - REST API framework

### Processing Pipeline

```
Input (mono WAV)
    ↓
1. Voice Isolation (Demucs) → extract vocals
    ↓
2. Noise Reduction (Neural) → remove static/hum
    ↓
3. Super-Resolution (AudioSR) → enhance to 48kHz
    ↓
4. Mono → Stereo conversion
    ↓
Output (MP3 stereo 192 kbps)
```

### Performance

**RTX 3090 benchmarks** (approximate):

- Voice Isolation: ~5 seconds per minute of audio
- Noise Reduction: ~1 second per minute
- Super-Resolution: ~10 seconds per minute
- **Total**: ~16 seconds per minute of audio

**Memory usage**: ~4-6GB VRAM

## 📁 Directory Structure

```
audio_enhancement_service/
├── audio_service.py          # Main service (Gradio + FastAPI)
├── requirements.txt          # Python dependencies
├── install.bat              # Installation script
├── start_service.bat        # Startup script
├── README.md                # This file
├── outputs/                 # Enhanced audio files (MP3)
└── temp/                    # Temporary processing files
```

## 🐛 Troubleshooting

### "CUDA out of memory" error

**Solution**: Reduce batch size or process shorter audio segments

### Demucs not found

**Solution**: Ensure virtual environment is activated:
```batch
call venv\Scripts\activate.bat
pip install demucs
```

### AudioSR model download fails

**Solution**: Download manually from Hugging Face:
- Model will auto-download on first use
- Requires internet connection

### FFmpeg not found (for MP3 conversion)

**Solution**: Install FFmpeg:
```batch
# Download from https://ffmpeg.org/download.html
# Or use Chocolatey:
choco install ffmpeg
```

## 📊 Output Quality

### Input vs Output

| Aspect | Input | Output |
|--------|-------|--------|
| Format | Mono WAV | **MP3 Stereo 192 kbps** |
| Sample Rate | Variable | **48,000 Hz** |
| Channels | 1 (mono) | **2 (stereo)** |
| Bitrate | N/A | **192 kbps** |
| Quality | Variable | **Broadcast quality** |

### Why MP3 Stereo 192 kbps?

- ✅ **Professional quality**: Indistinguishable from uncompressed
- ✅ **Universal compatibility**: Works everywhere
- ✅ **Reasonable file size**: ~1.4 MB per minute
- ✅ **Stereo**: More immersive, better for headphones

## 🔒 Privacy

- All processing happens **locally** on your RTX 3090
- **No data** is sent to external servers
- **No internet required** after initial model download

## 📜 License

MIT License - Free for personal and commercial use

## 🙏 Credits

- **AudioSR**: https://github.com/haoheliu/versatile_audio_super_resolution
- **Demucs**: https://github.com/facebookresearch/demucs
- **noisereduce**: https://github.com/timsainb/noisereduce
- **Gradio**: https://gradio.app
- **FastAPI**: https://fastapi.tiangolo.com

---

**Built with ❤️ for RTX 3090**
