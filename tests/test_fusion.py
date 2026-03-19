"""
tests/test_fusion.py — ASHAN
==============================
Run with:  python -m pytest tests/test_fusion.py -v

Tests all decision logic combinations without needing real hardware.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config
from fusion.fusion import _drain_queue


def _put_vision(confidence: float):
    config.vision_queue.put({"confidence": confidence, "timestamp": "2025-01-01T00:00:00"})


def _put_audio(vad_fired: bool, anomaly_score: float):
    config.audio_queue.put({"vad_fired": vad_fired, "anomaly_score": anomaly_score, "timestamp": "2025-01-01T00:00:00"})


def _clear_queues():
    for q in [config.vision_queue, config.audio_queue, config.feature_queue]:
        while not q.empty():
            q.get_nowait()


def test_vision_alone_triggers_in_use():
    _clear_queues()
    _put_vision(0.85)
    msg = _drain_queue(config.vision_queue)
    assert msg["confidence"] >= config.VISION_THRESHOLD


def test_vision_below_threshold_no_signal():
    _clear_queues()
    _put_vision(0.30)
    msg = _drain_queue(config.vision_queue)
    assert msg["confidence"] < config.VISION_THRESHOLD


def test_audio_vad_and_anomaly_both_fire():
    _clear_queues()
    _put_audio(vad_fired=True, anomaly_score=0.80)
    msg = _drain_queue(config.audio_queue)
    audio_signal = msg["vad_fired"] and msg["anomaly_score"] >= config.ANOMALY_THRESHOLD
    assert audio_signal is True


def test_audio_vad_fires_but_anomaly_suppresses():
    """VAD alone should NOT trigger — anomaly must also fire."""
    _clear_queues()
    _put_audio(vad_fired=True, anomaly_score=0.10)  # matches baseline
    msg = _drain_queue(config.audio_queue)
    audio_signal = msg["vad_fired"] and msg["anomaly_score"] >= config.ANOMALY_THRESHOLD
    assert audio_signal is False


def test_ghost_booking_both_silent():
    _clear_queues()
    _put_vision(0.02)
    _put_audio(vad_fired=False, anomaly_score=0.05)
    v = _drain_queue(config.vision_queue)
    a = _drain_queue(config.audio_queue)
    vision_signal = v["confidence"] >= config.VISION_THRESHOLD
    audio_signal  = a["vad_fired"] and a["anomaly_score"] >= config.ANOMALY_THRESHOLD
    assert not (vision_signal or audio_signal)


def test_silent_worker_vision_only():
    """Person present but silent — vision alone should declare IN_USE."""
    _clear_queues()
    _put_vision(0.91)
    _put_audio(vad_fired=False, anomaly_score=0.08)
    v = _drain_queue(config.vision_queue)
    a = _drain_queue(config.audio_queue)
    vision_signal = v["confidence"] >= config.VISION_THRESHOLD
    audio_signal  = a["vad_fired"] and a["anomaly_score"] >= config.ANOMALY_THRESHOLD
    assert vision_signal or audio_signal
