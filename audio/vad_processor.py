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
    log.info("[vad_processor] Importing PyTorch...")
    import torch
    log.info("[vad_processor] PyTorch %s imported OK (arch: %s)", torch.__version__, torch.get_default_dtype())
except ImportError as exc:
    log.error("[vad_processor] PyTorch import FAILED: %s", exc)
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
    log.info("[_init_vad] ── BEGIN VAD INIT ──")

    if torch is None:
        raise RuntimeError("PyTorch is not installed. Cannot start VAD workflow.")

    # Report quantization backend availability
    try:
        supported = list(torch.backends.quantized.supported_engines)
        log.info("[_init_vad] Quantized backends available: %s", supported)
        log.info("[_init_vad] Current quantized engine: %s", torch.backends.quantized.engine)
    except Exception as e:
        log.warning("[_init_vad] Could not query quantized backends: %s", e)

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_candidates = _get_model_candidates(base_dir)
    log.info("[_init_vad] Model search paths: %s", model_candidates)

    model_path = next((p for p in model_candidates if os.path.exists(p)), None)
    log.info("[_init_vad] Resolved model path: %s", model_path)

    if model_path is None:
        log.warning("[_init_vad] No VAD model found. Attempting to train/build automatically...")
        if _build_vad_model(base_dir):
            model_candidates = _get_model_candidates(base_dir)
            model_path = next((p for p in model_candidates if os.path.exists(p)), None)
            log.info("[_init_vad] After build, resolved model path: %s", model_path)

    if model_path is None:
        raise RuntimeError(
            "VAD model is unavailable after build attempt. "
            "Workflow is blocked until the model is built and loaded."
        )

    try:
        log.info("[_init_vad] Loading TorchScript model from %s ...", model_path)
        _model = torch.jit.load(model_path)
        log.info("[_init_vad] torch.jit.load() succeeded.")
        _model.eval()
        log.info("[_init_vad] model.eval() done.")

        # Smoke-test: run one dummy inference to catch backend mismatches early
        log.info("[_init_vad] Running smoke-test inference (512 samples @ 16kHz)...")
        dummy = torch.zeros(1, 512)
        result = _model(dummy, 16000)
        log.info("[_init_vad] Smoke-test result: %s  ── VAD INIT OK ──", result)

    except Exception as e:
        log.error("[_init_vad] Could not load %s: %s", model_path, e, exc_info=True)
        log.warning("[_init_vad] Retrying with a fresh build...")
        if _build_vad_model(base_dir):
            model_candidates = _get_model_candidates(base_dir)
            model_path = next((p for p in model_candidates if os.path.exists(p)), None)
            if model_path is not None:
                try:
                    log.info("[_init_vad] Retrying torch.jit.load(%s)...", model_path)
                    _model = torch.jit.load(model_path)
                    _model.eval()
                    log.info("[_init_vad] Retry succeeded  ── VAD INIT OK ──")
                    return
                except Exception as e2:
                    log.error("[_init_vad] Retry also failed: %s", e2, exc_info=True)
        raise RuntimeError(
            "VAD model load failed after rebuild. "
            "Workflow is blocked until model loads successfully."
        ) from e

def _init_audio_stream():
    """Initialise PyAudio input stream. Called once."""
    global _stream, _pyaudio
    log.info("[_init_audio_stream] ── BEGIN AUDIO STREAM INIT ──")
    try:
        log.info("[_init_audio_stream] Importing pyaudio...")
        import pyaudio
        log.info("[_init_audio_stream] pyaudio imported OK.")

        log.info("[_init_audio_stream] Creating PyAudio instance...")
        _pyaudio = pyaudio.PyAudio()
        log.info("[_init_audio_stream] PyAudio instance created.")

        # List available devices for diagnostics
        dev_count = _pyaudio.get_device_count()
        log.info("[_init_audio_stream] Found %d audio devices:", dev_count)
        for i in range(dev_count):
            try:
                info = _pyaudio.get_device_info_by_index(i)
                log.info("  [%d] %s  (inputs=%d, rate=%.0f)",
                         i, info['name'], info['maxInputChannels'], info['defaultSampleRate'])
            except Exception:
                log.info("  [%d] <could not query>", i)

        frame_samples = int(config.AUDIO_SAMPLE_RATE * config.AUDIO_CHUNK_MS / 1000)
        log.info("[_init_audio_stream] Opening stream: rate=%d, chunk_ms=%d, frame_samples=%d, device_index=%s",
                 config.AUDIO_SAMPLE_RATE, config.AUDIO_CHUNK_MS, frame_samples, config.AUDIO_DEVICE_INDEX)

        _stream = _pyaudio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=config.AUDIO_SAMPLE_RATE,
            input=True,
            input_device_index=config.AUDIO_DEVICE_INDEX,
            frames_per_buffer=frame_samples,
        )
        log.info("[_init_audio_stream] Audio stream opened OK  ── AUDIO STREAM INIT DONE ──")
    except Exception as e:
        log.error("[_init_audio_stream] Failed to open audio stream: %s", e, exc_info=True)


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
        log.info("[is_speech] VAD model not loaded yet — calling _init_vad()...")
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
        log.error("[is_speech] Silero VAD inference error: %s", e, exc_info=True)
        return False
