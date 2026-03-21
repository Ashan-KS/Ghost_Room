"""
audio/audio_loop.py — RAHUL
=============================
Single capture loop that drives both VAD and Anomaly detection.

One mic stream → captures a 30ms chunk each iteration:
  1. Runs WebRTC VAD        → writes to audio_queue
  2. Extracts features      → feeds to anomaly model
  3. Gets anomaly score     → writes to anomaly_queue

Fusion reads the latest value from each queue independently.
"""

import logging
import time
from datetime import datetime, timezone

import numpy as np

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import app_config
from audio.vad_processor     import is_speech, get_audio_chunk
from audio.feature_extractor import extract_features
from anomaly.anomaly_model    import get_anomaly_score, train_and_save, is_calibrated

log = logging.getLogger(__name__)


def audio_loop():
    """Main audio thread — runs forever."""
    log.info("Audio loop started.")
    log.info(
        f"  Sample rate={app_config.AUDIO_SAMPLE_RATE}Hz  "
        f"chunk={app_config.AUDIO_CHUNK_MS}ms  "
        f"VAD mode={app_config.VAD_MODE}"
    )

    frame_count = 0
    speech_count = 0
    anomaly_count = 0

    while True:
        try:
            # ── 1. Capture ────────────────────────────────────────────────
            chunk = get_audio_chunk()
            if chunk is None:
                time.sleep(0.01)
                continue

            frame_count += 1
            timestamp = datetime.now(timezone.utc).isoformat()

            # ── 2. VAD ────────────────────────────────────────────────────
            vad_fired = is_speech(chunk)
            app_config.audio_queue.put({
                "vad_fired": vad_fired,
                "timestamp": timestamp,
            })

            if vad_fired:
                speech_count += 1

            # ── 3. Anomaly ────────────────────────────────────────────────
            feature_vector = extract_features(chunk)
            
            if is_calibrated():
                anomaly_score = get_anomaly_score(feature_vector)
                app_config.anomaly_queue.put({
                    "anomaly_score": float(anomaly_score),
                    "timestamp":     timestamp,
                })
            else:
                anomaly_score = 0.0
                # Optionally warn once or periodically
                if frame_count % 100 == 1:
                    log.warning("Anomaly model not calibrated — skipping inference.")

            if anomaly_score >= app_config.ANOMALY_THRESHOLD:
                anomaly_count += 1

            # ── 4. Periodic stats ───────────────────────────────────────────
            frames_per_interval = int((app_config.AUDIO_LOG_STATS_INTERVAL_S * 1000) / app_config.AUDIO_CHUNK_MS)
            if frame_count % frames_per_interval == 0:
                log.info(
                    f"Audio stats ({frame_count} frames): "
                    f"speech={speech_count}  "
                    f"anomaly={anomaly_count}  "
                    f"latest_vad={vad_fired}  "
                    f"latest_anom={anomaly_score:.1f}"
                )

        except Exception as e:
            log.error(f"Audio loop error: {e}", exc_info=True)
            time.sleep(0.1)
