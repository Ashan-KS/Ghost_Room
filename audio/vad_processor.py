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
import numpy as np
import config

log = logging.getLogger(__name__)

_vad      = None
_stream   = None
_pyaudio  = None

VAD_MODE = 2  # 0=least aggressive, 3=most aggressive


def _init_vad():
    """Initialise WebRTC VAD. Called once."""
    global _vad
    try:
        import webrtcvad
        _vad = webrtcvad.Vad(VAD_MODE)
        log.info(f"WebRTC VAD initialised (mode={VAD_MODE})")
    except ImportError:
        log.warning("webrtcvad not installed — is_speech() will always return False.")


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
    Run WebRTC VAD on a raw PCM chunk.

    Args:
        chunk: bytes — raw int16 PCM at AUDIO_SAMPLE_RATE

    Returns:
        bool: True if speech detected.
    """
    global _vad

    if _vad is None:
        _init_vad()

    if _vad is None:
        return False

    try:
        return _vad.is_speech(chunk, config.AUDIO_SAMPLE_RATE)
    except Exception as e:
        log.error(f"VAD error: {e}")
        return False
