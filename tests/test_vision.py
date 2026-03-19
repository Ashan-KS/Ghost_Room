"""
tests/test_vision.py — SACHITH
================================
Run with:  python -m pytest tests/test_vision.py -v

Tests you must pass before pushing:
  1. get_camera_frame() returns a numpy array of the right shape
  2. preprocess_frame() output is (1, 300, 300, 3) uint8
  3. run_inference() returns a float between 0.0 and 1.0
  4. vision_queue receives a correctly shaped message
"""

import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_preprocess_frame_shape():
    from vision.frame_utils import preprocess_frame
    fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = preprocess_frame(fake_frame)
    assert result.shape == (1, 300, 300, 3), f"Expected (1,300,300,3), got {result.shape}"
    assert result.dtype == np.uint8


def test_preprocess_frame_range():
    from vision.frame_utils import preprocess_frame
    fake_frame = (np.random.rand(480, 640, 3) * 255).astype(np.uint8)
    result = preprocess_frame(fake_frame)
    assert result.min() >= 0 and result.max() <= 255


def test_run_inference_returns_float():
    """Smoke test — just checks the return type and range."""
    from vision.vision_inference import run_inference
    fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    score = run_inference(fake_frame)
    assert isinstance(score, float), f"Expected float, got {type(score)}"
    assert 0.0 <= score <= 1.0, f"Score {score} out of range [0, 1]"


def test_vision_queue_message_shape():
    """Simulate one camera loop tick and check queue message contract."""
    import config
    from datetime import datetime, timezone
    from vision.vision_inference import run_inference
    from vision.frame_utils import get_camera_frame, preprocess_frame

    # Clear queue
    while not config.vision_queue.empty():
        config.vision_queue.get_nowait()

    frame = get_camera_frame()
    assert frame is not None, "Camera returned None — is your webcam connected?"

    confidence = run_inference(frame)
    message = {
        "confidence": confidence,
        "timestamp":  datetime.now(timezone.utc).isoformat(),
    }
    config.vision_queue.put(message)

    result = config.vision_queue.get_nowait()
    assert "confidence" in result, "Message missing 'confidence' key"
    assert "timestamp"  in result, "Message missing 'timestamp' key"
    assert isinstance(result["confidence"], float)
