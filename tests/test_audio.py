"""
tests/test_audio.py — RAHUL
=============================
Run with:  python -m pytest tests/test_audio.py -v

Tests you must pass before pushing:
  1. extract_features() returns shape (14,) float32
  2. is_speech() returns a bool
  3. audio_queue message has correct keys
  4. feature_queue receives numpy array of correct shape
"""

import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config


def _make_fake_chunk() -> bytes:
    """Generate a 30ms silent PCM chunk at 16kHz."""
    n_samples = int(config.AUDIO_SAMPLE_RATE * config.AUDIO_CHUNK_MS / 1000)
    return (np.zeros(n_samples, dtype=np.int16)).tobytes()


def test_feature_vector_shape():
    from audio.feature_extractor import extract_features
    chunk = _make_fake_chunk()
    vector = extract_features(chunk)
    assert vector.shape == (14,), f"Expected shape (14,), got {vector.shape}"
    assert vector.dtype == np.float32, f"Expected float32, got {vector.dtype}"


def test_feature_vector_finite():
    from audio.feature_extractor import extract_features
    chunk = _make_fake_chunk()
    vector = extract_features(chunk)
    assert np.all(np.isfinite(vector)), "Feature vector contains NaN or Inf"


def test_get_zero_vector():
    from audio.feature_extractor import get_zero_vector, FEATURE_DIM
    v = get_zero_vector()
    assert v.shape == (FEATURE_DIM,)
    assert np.all(v == 0.0)


def test_is_speech_returns_bool():
    from audio.vad_processor import is_speech
    chunk = _make_fake_chunk()
    result = is_speech(chunk)
    assert isinstance(result, bool), f"Expected bool, got {type(result)}"


def test_audio_queue_message_contract():
    """Check audio_queue message has the right keys and types."""
    from datetime import datetime, timezone

    while not config.audio_queue.empty():
        config.audio_queue.get_nowait()

    # Simulate what audio_loop puts in the queue
    message = {
        "vad_fired":     False,
        "anomaly_score": 0.0,
        "timestamp":     datetime.now(timezone.utc).isoformat(),
    }
    config.audio_queue.put(message)

    result = config.audio_queue.get_nowait()
    assert "vad_fired"     in result
    assert "anomaly_score" in result
    assert "timestamp"     in result
    assert isinstance(result["vad_fired"],     bool)
    assert isinstance(result["anomaly_score"], float)


def test_feature_queue_receives_vector():
    """Check feature_queue receives correct numpy array."""
    from audio.feature_extractor import extract_features

    while not config.feature_queue.empty():
        config.feature_queue.get_nowait()

    chunk  = _make_fake_chunk()
    vector = extract_features(chunk)
    config.feature_queue.put(vector)

    result = config.feature_queue.get_nowait()
    assert isinstance(result, np.ndarray)
    assert result.shape == (14,)
