"""
anomaly/calibration.py — GINURA
=================================
Records ambient sensor data from an empty room and trains the IsolationForest.
Called once at startup if no saved baseline exists.

Uses the anomaly model's own 28-dim feature extractor
(extract_anomaly_features) so that calibration and runtime inference
operate on the exact same feature space.

Depends on: audio/vad_processor.py (Rahul's module) for mic capture.
"""

import logging
import time
import numpy as np

import config
from audio.vad_processor     import get_audio_chunk
from anomaly.anomly_model    import train_and_save, extract_anomaly_features

log = logging.getLogger(__name__)


def run_calibration(duration_s=None, progress_callback=None):
    """
    Collect CALIBRATION_DURATION seconds of ambient audio features
    and train the IsolationForest baseline model.

    Args:
        duration_s: Override for calibration duration in seconds.
        progress_callback: Optional callable(progress_int, message_str)
                           called on each progress update so callers
                           (e.g. MQTT publisher) can stream progress.

    Blocks until complete. Call this before spawning threads.
    """
    if duration_s is None:
        duration_s = config.CALIBRATION_DURATION_S
    log.info(f"Calibration: collecting {duration_s}s of ambient data...")
    log.info("Make sure the room is EMPTY and quiet during this phase.")

    if progress_callback:
        progress_callback(0, f"Starting calibration for {duration_s}s...")

    feature_matrix = []
    start_time     = time.time()
    chunk_duration = config.AUDIO_CHUNK_MS / 1000.0

    while (time.time() - start_time) < duration_s:
        try:
            chunk = get_audio_chunk()
            if chunk is not None:
                # Convert raw int16 PCM bytes → float32 samples normalised to [-1, 1]
                samples = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
                vector = extract_anomaly_features(samples)
                feature_matrix.append(vector)
        except Exception as e:
            log.error(f"Calibration capture error: {e}")
            time.sleep(chunk_duration)

        elapsed  = time.time() - start_time
        progress = int((elapsed / duration_s) * 100)
        if len(feature_matrix) % 100 == 0:
            log.info(f"Calibration progress: {progress}% ({len(feature_matrix)} samples)")
            if progress_callback:
                progress_callback(progress, f"{progress}% — {len(feature_matrix)} samples collected")

    X = np.array(feature_matrix)
    log.info(f"Calibration collected {X.shape[0]} samples, shape {X.shape}")
    if progress_callback:
        progress_callback(90, f"Training model on {X.shape[0]} samples...")

    train_and_save(X)
    log.info("Calibration complete — baseline.pkl saved.")
    if progress_callback:
        progress_callback(100, "Calibration complete — baseline.pkl saved.")


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
    duration = int(sys.argv[1]) if len(sys.argv) > 1 else None
    run_calibration(duration)
