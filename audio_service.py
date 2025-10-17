"""
Audio Enhancement Service - Hybrid Gradio + FastAPI
Optimized for RTX 3090
Combines Voice Isolation (Demucs) + Audio Super-Resolution (AudioSR)
"""

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Tuple
import warnings
warnings.filterwarnings("ignore")

import torch
import numpy as np
import soundfile as sf
import noisereduce as nr
from pydub import AudioSegment
import gradio as gr
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
import uvicorn

# Check GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")

# Create output directories
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)
TEMP_DIR = Path("temp")
TEMP_DIR.mkdir(exist_ok=True)

# Global model cache
audiosr_model = None

def load_audiosr():
    """Lazy loading of AudioSR model"""
    global audiosr_model
    if audiosr_model is None:
        print("Loading AudioSR model...")
        import audiosr
        audiosr_model = audiosr.build_model(model_name="basic", device=device)
        print("AudioSR model loaded!")
    return audiosr_model

def isolate_voice_demucs(audio_path: str) -> str:
    """
    Step 1: Voice isolation using Demucs v4
    Returns path to isolated vocals
    """
    print(f"Step 1/3: Isolating voice with Demucs...")

    output_dir = TEMP_DIR / "demucs_output"
    output_dir.mkdir(exist_ok=True)

    # Run Demucs for vocal separation
    cmd = [
        "demucs",
        "--two-stems=vocals",
        "-o", str(output_dir),
        "--device", device,
        audio_path
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"Demucs error: {e.stderr}")
        raise RuntimeError(f"Demucs failed: {e.stderr}")

    # Find the vocals file
    audio_name = Path(audio_path).stem
    vocals_path = output_dir / "htdemucs" / audio_name / "vocals.wav"

    if not vocals_path.exists():
        raise FileNotFoundError(f"Vocals not found at {vocals_path}")

    print(f"Voice isolated: {vocals_path}")
    return str(vocals_path)

def reduce_noise(audio_path: str) -> Tuple[np.ndarray, int]:
    """
    Step 2: Noise reduction using neural network
    Returns cleaned audio array and sample rate
    """
    print(f"Step 2/3: Reducing noise...")

    audio, sr = sf.read(audio_path)

    # Apply neural noise reduction
    cleaned_audio = nr.reduce_noise(
        y=audio,
        sr=sr,
        stationary=True,
        prop_decrease=0.8
    )

    print(f"Noise reduced (sample rate: {sr} Hz)")
    return cleaned_audio, sr

def enhance_audio_sr(audio: np.ndarray, sr: int, target_sr: int = 48000) -> Tuple[np.ndarray, int]:
    """
    Step 3: Audio super-resolution to enhance quality
    Returns enhanced audio at target sample rate
    """
    print(f"Step 3/3: Enhancing quality with AudioSR (target: {target_sr} Hz)...")

    model = load_audiosr()

    # Convert to mono if stereo
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)

    # Normalize
    audio = audio / (np.max(np.abs(audio)) + 1e-8)

    # Save temp file for AudioSR
    temp_file = TEMP_DIR / "temp_for_sr.wav"
    sf.write(temp_file, audio, sr)

    # AudioSR enhancement
    enhanced = model(
        str(temp_file),
        guidance_scale=3.5,
        ddim_steps=50,
        seed=42
    )

    # AudioSR returns tensor, convert to numpy
    if isinstance(enhanced, torch.Tensor):
        enhanced = enhanced.cpu().numpy()

    # Squeeze to remove batch dimension
    if len(enhanced.shape) > 1:
        enhanced = enhanced.squeeze()

    print(f"Audio enhanced to {target_sr} Hz")
    return enhanced, target_sr

def convert_to_stereo(audio: np.ndarray) -> np.ndarray:
    """
    Convert mono audio to stereo by duplicating channel
    """
    if len(audio.shape) == 1:
        # Mono to stereo: duplicate channel
        stereo = np.stack([audio, audio], axis=1)
        return stereo
    return audio

def save_as_mp3(audio: np.ndarray, sr: int, output_path: str, bitrate: str = "192k"):
    """
    Save audio as MP3 stereo 192 kbps

    Args:
        audio: Audio array (mono or stereo)
        sr: Sample rate
        output_path: Output file path (.mp3)
        bitrate: MP3 bitrate (default: 192k)
    """
    # Convert to stereo if mono
    stereo_audio = convert_to_stereo(audio)

    # Save as temporary WAV first
    temp_wav = TEMP_DIR / "temp_for_mp3.wav"
    sf.write(temp_wav, stereo_audio, sr)

    # Convert to MP3 using pydub
    audio_segment = AudioSegment.from_wav(temp_wav)
    audio_segment.export(
        output_path,
        format="mp3",
        bitrate=bitrate,
        parameters=["-ac", "2"]  # Force stereo
    )

    print(f"Saved as MP3 stereo {bitrate}: {output_path}")

def process_audio(
    input_file: str,
    enable_demucs: bool = True,
    enable_noise_reduction: bool = True,
    enable_sr: bool = True,
    target_sr: int = 48000
) -> str:
    """
    Full audio enhancement pipeline

    Args:
        input_file: Path to input audio file
        enable_demucs: Enable voice isolation
        enable_noise_reduction: Enable noise reduction
        enable_sr: Enable super-resolution
        target_sr: Target sample rate for super-resolution

    Returns:
        Path to enhanced audio file
    """
    try:
        current_file = input_file

        # Step 1: Voice isolation (optional)
        if enable_demucs:
            current_file = isolate_voice_demucs(current_file)

        # Step 2: Noise reduction (optional)
        if enable_noise_reduction:
            audio, sr = reduce_noise(current_file)
            temp_cleaned = TEMP_DIR / "cleaned.wav"
            sf.write(temp_cleaned, audio, sr)
            current_file = str(temp_cleaned)
        else:
            audio, sr = sf.read(current_file)

        # Step 3: Super-resolution (optional)
        if enable_sr:
            if not enable_noise_reduction:
                audio, sr = sf.read(current_file)
            enhanced_audio, enhanced_sr = enhance_audio_sr(audio, sr, target_sr)
        else:
            enhanced_audio, enhanced_sr = audio, sr

        # Save final output as MP3 stereo 192 kbps
        output_filename = f"enhanced_{Path(input_file).stem}.mp3"
        output_path = OUTPUT_DIR / output_filename

        save_as_mp3(enhanced_audio, enhanced_sr, str(output_path), bitrate="192k")

        print(f"Processing complete! Output: {output_path}")
        return str(output_path)

    except Exception as e:
        print(f"Error during processing: {e}")
        raise

# ============================================================================
# GRADIO INTERFACE
# ============================================================================

def gradio_process(
    audio_file,
    use_demucs: bool,
    use_noise_reduction: bool,
    use_sr: bool,
    target_sample_rate: int
):
    """Gradio wrapper for audio processing"""
    if audio_file is None:
        return None, "Please upload an audio file"

    try:
        output_path = process_audio(
            audio_file,
            enable_demucs=use_demucs,
            enable_noise_reduction=use_noise_reduction,
            enable_sr=use_sr,
            target_sr=target_sample_rate
        )

        return output_path, f"Success! Enhanced audio saved to: {output_path}"

    except Exception as e:
        return None, f"Error: {str(e)}"

def create_gradio_interface():
    """Create Gradio web interface"""
    with gr.Blocks(title="Audio Enhancement Service") as demo:
        gr.Markdown("""
        # 🎵 Audio Enhancement Service
        **AI-Powered Voice Isolation + Quality Enhancement**

        Combines Demucs (voice isolation) + noisereduce + AudioSR (super-resolution)
        Optimized for RTX 3090 | CUDA 12.4
        """)

        with gr.Row():
            with gr.Column():
                audio_input = gr.Audio(
                    label="Upload Audio File",
                    type="filepath"
                )

                gr.Markdown("### Enhancement Options")

                demucs_check = gr.Checkbox(
                    label="Voice Isolation (Demucs v4)",
                    value=True,
                    info="Separate vocals from background"
                )

                noise_check = gr.Checkbox(
                    label="Noise Reduction (Neural)",
                    value=True,
                    info="Remove background noise"
                )

                sr_check = gr.Checkbox(
                    label="Audio Super-Resolution (AudioSR)",
                    value=True,
                    info="Enhance quality and upscale"
                )

                target_sr_slider = gr.Slider(
                    minimum=16000,
                    maximum=48000,
                    step=8000,
                    value=48000,
                    label="Target Sample Rate (Hz)",
                    info="Higher = better quality, slower processing"
                )

                process_btn = gr.Button("🚀 Enhance Audio", variant="primary")

            with gr.Column():
                audio_output = gr.Audio(
                    label="Enhanced Audio Output"
                )

                status_text = gr.Textbox(
                    label="Status",
                    lines=3
                )

        gr.Markdown("""
        ### 💡 Tips
        - **Voice Isolation**: Best for removing music/noise from speech
        - **Noise Reduction**: Removes static, hum, background noise
        - **Super-Resolution**: Upscales to studio quality (48kHz)
        - Use all three for maximum quality improvement

        **Output Format**: MP3 Stereo 192 kbps (mono input → stereo output)
        """)

        process_btn.click(
            fn=gradio_process,
            inputs=[audio_input, demucs_check, noise_check, sr_check, target_sr_slider],
            outputs=[audio_output, status_text]
        )

    return demo

# ============================================================================
# FASTAPI INTERFACE
# ============================================================================

app = FastAPI(title="Audio Enhancement API", version="1.0.0")

@app.get("/")
def read_root():
    """API status endpoint"""
    return {
        "service": "Audio Enhancement API",
        "version": "1.0.0",
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else "N/A",
        "endpoints": {
            "POST /enhance": "Upload audio file for enhancement",
            "GET /outputs/{filename}": "Download enhanced audio"
        }
    }

@app.post("/enhance")
async def enhance_audio_api(
    file: UploadFile = File(...),
    enable_demucs: bool = True,
    enable_noise_reduction: bool = True,
    enable_sr: bool = True,
    target_sr: int = 48000
):
    """
    API endpoint for audio enhancement

    Upload an audio file and get enhanced version
    """
    # Save uploaded file
    temp_input = TEMP_DIR / f"upload_{file.filename}"
    with open(temp_input, "wb") as f:
        content = await file.read()
        f.write(content)

    try:
        output_path = process_audio(
            str(temp_input),
            enable_demucs=enable_demucs,
            enable_noise_reduction=enable_noise_reduction,
            enable_sr=enable_sr,
            target_sr=target_sr
        )

        return {
            "status": "success",
            "output_file": Path(output_path).name,
            "download_url": f"/outputs/{Path(output_path).name}"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/outputs/{filename}")
def download_output(filename: str):
    """Download enhanced audio file"""
    file_path = OUTPUT_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(
        path=file_path,
        media_type="audio/mpeg",
        filename=filename
    )

# ============================================================================
# MAIN - HYBRID SERVER
# ============================================================================

def main():
    """Launch hybrid Gradio + FastAPI server"""
    import threading

    print("="*80)
    print("🎵 AUDIO ENHANCEMENT SERVICE - HYBRID MODE")
    print("="*80)
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA: {torch.version.cuda}")
    print("="*80)

    # Create Gradio interface
    gradio_app = create_gradio_interface()

    # Mount FastAPI app within Gradio
    gradio_app.queue()

    # Launch hybrid server
    print("\n🚀 Starting services...")
    print("\n📊 Gradio Interface: http://localhost:7860")
    print("🔌 FastAPI Docs: http://localhost:7860/docs")
    print("\n" + "="*80 + "\n")

    # Launch with FastAPI mounted
    gr.mount_gradio_app(app, gradio_app, path="/")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=7860,
        log_level="info"
    )

if __name__ == "__main__":
    main()
