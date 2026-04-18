"""
audio/vad_processor.py — RAHUL
================================
WebRTC VAD wrapper and audio capture.

Your tasks:
  1. Implement get_audio_chunk() — captures one 30ms frame from mic
  2. Implement is_speech() — returns True/False from webrtcvad
  3. Handle hardware abstraction (laptop mic vs USB mic on Pi)

Install: pip install webrtcvad pyaudio
"""

import logging
# import numpy as np
import config
import struct
import os
import subprocess
import sys

# We need PyTorch for Edge AI inferencing
try:
    import torch
except ImportError:
    torch = None

log = logging.getLogger(__name__)

_model    = None
_stream   = None
_pyaudio  = None


def _get_model_candidates(base_dir: str) -> list[str]:
    audio_dir = os.path.dirname(os.path.abspath(__file__))
    return [
        os.path.join(base_dir, "models", "silero_vad_int8.pt"),
        os.path.join(audio_dir, "models", "silero_vad_int8.pt"),
        os.path.join("models", "silero_vad_int8.pt"),
    ]


def _build_vad_model(base_dir: str) -> bool:
    """Run the VAD training/quantization script to generate a loadable model."""
    train_script = os.path.join(base_dir, "audio", "vad_model.py")
    if not os.path.exists(train_script):
        log.error("VAD training script not found at %s", train_script)
        return False

    cmd = [sys.executable, train_script]
    log.info("Building VAD model with command: %s", " ".join(cmd))

    try:
        subprocess.run(cmd, cwd=base_dir, check=True)
        return True
    except subprocess.CalledProcessError as e:
        log.error("VAD model build failed (exit=%s)", e.returncode)
        return False
    except Exception as e:
        log.error("Unexpected error while building VAD model: %s", e)
        return False


def _get_vad_frame_samples(sample_rate: int) -> int:
    if sample_rate == 16000:
        return 512
    if sample_rate == 8000:
        return 256
    raise ValueError(f"Unsupported sample rate for Silero VAD: {sample_rate}")


def _prepare_for_silero(chunk: bytes):
    """Convert raw int16 chunk to normalized tensor with valid Silero frame length."""
    num_samples = len(chunk) // 2
    audio_int16 = struct.unpack(f"{num_samples}h", chunk)
    audio_float32 = [x / 32768.0 for x in audio_int16]

    expected = _get_vad_frame_samples(config.AUDIO_SAMPLE_RATE)
    if len(audio_float32) < expected:
        audio_float32.extend([0.0] * (expected - len(audio_float32)))
    elif len(audio_float32) > expected:
        audio_float32 = audio_float32[:expected]

    return torch.tensor([audio_float32], dtype=torch.float32)



def _init_vad():
    """Initialise PyTorch Silero VAD (INT8 optimized). Called once."""
    global _model
    
    if torch is None:
        raise RuntimeError("PyTorch is not installed. Cannot start VAD workflow.")

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_candidates = _get_model_candidates(base_dir)
    model_path = next((p for p in model_candidates if os.path.exists(p)), None)

    if model_path is None:
        log.warning("No VAD model found. Attempting to train/build automatically...")
        if _build_vad_model(base_dir):
            model_candidates = _get_model_candidates(base_dir)
            model_path = next((p for p in model_candidates if os.path.exists(p)), None)

    if model_path is None:
        raise RuntimeError(
            "VAD model is unavailable after build attempt. "
            "Workflow is blocked until the model is built and loaded."
        )

    try:
        # Load the optimized TorchScript model
        _model = torch.jit.load(model_path)
        _model.eval()
        log.info(f"Edge AI Silero VAD loaded from {model_path}")
    except Exception as e:
        log.warning(f"Could not load {model_path}: {e}. Retrying with a fresh build...")
        if _build_vad_model(base_dir):
            model_candidates = _get_model_candidates(base_dir)
            model_path = next((p for p in model_candidates if os.path.exists(p)), None)
            if model_path is not None:
                try:
                    _model = torch.jit.load(model_path)
                    _model.eval()
                    log.info(f"Edge AI Silero VAD loaded from {model_path}")
                    return
                except Exception as e2:
                    log.error("Could not load rebuilt VAD model at %s: %s", model_path, e2)
        raise RuntimeError(
            "VAD model load failed after rebuild. "
            "Workflow is blocked until model loads successfully."
        ) from e

def _init_audio_stream():
    """Initialise PyAudio input stream. Called once."""
    global _stream, _pyaudio
    try:
        import pyaudio
        _pyaudio = pyaudio.PyAudio()
        frame_samples = int(config.AUDIO_SAMPLE_RATE * config.AUDIO_CHUNK_MS / 1000)
        _stream = _pyaudio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=config.AUDIO_SAMPLE_RATE,
            input=True,
            input_device_index=config.AUDIO_DEVICE_INDEX,
            frames_per_buffer=frame_samples,
        )
        log.info("Audio stream opened.")
    except Exception as e:
        log.error(f"Failed to open audio stream: {e}")


def get_audio_chunk() -> bytes | None:
    """
    Capture one audio chunk (AUDIO_CHUNK_MS milliseconds).

    Returns:
        bytes: raw PCM int16 audio, or None on failure.
    """
    global _stream

    if _stream is None:
        _init_audio_stream()

    if _stream is None:
        return None

    try:
        frame_samples = int(config.AUDIO_SAMPLE_RATE * config.AUDIO_CHUNK_MS / 1000)
        return _stream.read(frame_samples, exception_on_overflow=False)
    except Exception as e:
        log.error(f"Error capturing audio chunk: {e}")
        return None


def is_speech(chunk: bytes) -> bool:
    """
    Run PyTorch INT8 Silero VAD on a raw PCM chunk.

    Args:
        chunk: bytes — raw int16 PCM at AUDIO_SAMPLE_RATE

    Returns:
        bool: True if speech detected above config.VAD_THRESHOLD.
    """
    global _model

    if _model is None:
        _init_vad()

    if _model is None or torch is None:
        raise RuntimeError("VAD model is not ready. Workflow cannot continue.")

    try:
        tensor_chunk = _prepare_for_silero(chunk)
        
        # Inference using the compiled model
        # Using configured sample rate
        with torch.no_grad():
            confidence = _model(tensor_chunk, config.AUDIO_SAMPLE_RATE).item()
            
        return confidence > getattr(config, 'VAD_THRESHOLD', 0.5)

    except Exception as e:
        log.error(f"Silero VAD error: {e}")
        return False
