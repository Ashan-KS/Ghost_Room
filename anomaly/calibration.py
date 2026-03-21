"""
anomaly/calibration.py — GINURA
=================================
Records ambient sensor data from an empty room and trains the IsolationForest.
Called once at startup if no saved baseline exists.

Your tasks:
  1. Implement run_calibration() — collects features for CALIBRATION_DURATION seconds
  2. Call anomaly_model.train_and_save() with the collected feature matrix
  3. Test on your laptop: record 5 min of ambient audio in a quiet room,
     then introduce noise (talk, clap) and verify anomaly score rises above threshold.

Depends on: audio/feature_extractor.py (Rahul's module)
  — Make sure Rahul's extract_features() is working before you test calibration.
  — Use get_zero_vector() as a stub if Rahul isn't ready yet.
"""

import logging
import time
import numpy as np

import sys
import os

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app_config
from audio.feature_extractor import extract_features, get_zero_vector, FEATURE_DIM
from audio.vad_processor     import get_audio_chunk
from anomaly.anomaly_model    import train_and_save

log = logging.getLogger(__name__)


def run_calibration():
    """
    Collect CALIBRATION_DURATION seconds of ambient audio features
    and train the IsolationForest baseline model.

    Blocks until complete. Call this before spawning threads.
    """
    log.info(f"Calibration: collecting {app_config.CALIBRATION_DURATION_S}s of ambient data...")
    log.info("Make sure the room is EMPTY and quiet during this phase.")

    feature_matrix = []
    start_time     = time.time()
    chunk_duration = app_config.AUDIO_CHUNK_MS / 1000.0

    while (time.time() - start_time) < app_config.CALIBRATION_DURATION_S:
        try:
            chunk = get_audio_chunk()
            if chunk is not None:
                vector = extract_features(chunk)
                feature_matrix.append(vector)
        except NotImplementedError:
            # Rahul's code not ready — use zero vectors as placeholder
            log.warning("extract_features() not implemented — using zero vectors (placeholder).")
            feature_matrix.append(get_zero_vector())
            time.sleep(chunk_duration)
        except Exception as e:
            log.error(f"Calibration capture error: {e}")
            time.sleep(chunk_duration)

        elapsed  = time.time() - start_time
        progress = int((elapsed / app_config.CALIBRATION_DURATION_S) * 100)
        if len(feature_matrix) % 100 == 0:
            log.info(f"Calibration progress: {progress}% ({len(feature_matrix)} samples)")

    X = np.array(feature_matrix)
    log.info(f"Calibration collected {X.shape[0]} samples, shape {X.shape}")

    # TODO (Ginura): call train_and_save with the collected matrix
    train_and_save(X)
    log.info("Calibration complete — baseline.pkl saved.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
    run_calibration()
