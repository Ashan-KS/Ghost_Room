"""
anomaly/anomaly_model.py — GINURA
====================================
IsolationForest wrapper — training, saving, loading, and inference.

Your tasks:
  1. Implement train_and_save() — fits the model on calibration data, saves to disk
  2. Implement load_model() — loads saved model from disk on reboot
  3. Implement get_anomaly_score() — returns normalised float 0.0–1.0

Interface contract with Rahul (do not change):
  get_anomaly_score(vector: np.ndarray shape (14,)) → float 0.0–1.0

Interface contract with Ashan (do not change):
  get_anomaly_score() returns a float — Ashan thresholds it at ANOMALY_THRESHOLD

Install: pip install scikit-learn joblib
"""

import logging
import numpy as np
import config

log  = logging.getLogger(__name__)
_model = None   # loaded IsolationForest instance


def train_and_save(X: np.ndarray):
    """
    Fit IsolationForest on calibration feature matrix and save to disk.

    Args:
        X: numpy array shape (n_samples, 14) — collected during calibration
    """
    global _model

    # TODO (Ginura): implement this
    # Steps:
    #   from sklearn.ensemble import IsolationForest
    #   import joblib
    #   _model = IsolationForest(contamination=0.05, random_state=42)
    #   _model.fit(X)
    #   joblib.dump(_model, config.ANOMALY_MODEL_PATH)
    #   log.info(f"Model trained on {X.shape[0]} samples, saved to {config.ANOMALY_MODEL_PATH}")

    raise NotImplementedError("Ginura: implement train_and_save() in anomaly_model.py")


def load_model():
    """Load a previously saved model from disk."""
    global _model

    # TODO (Ginura): implement this
    # import joblib
    # _model = joblib.load(config.ANOMALY_MODEL_PATH)
    # log.info(f"Baseline model loaded from {config.ANOMALY_MODEL_PATH}")

    raise NotImplementedError("Ginura: implement load_model() in anomaly_model.py")


def get_anomaly_score(vector: np.ndarray) -> float:
    """
    Score a single feature vector against the trained baseline.

    Args:
        vector: numpy array shape (14,) — from Rahul's extract_features()

    Returns:
        float: anomaly score 0.0–1.0.
               0.0 = identical to baseline (definitely empty).
               1.0 = completely unlike baseline (something is happening).

    IsolationForest.decision_function() returns:
        positive values → normal (matches baseline)
        negative values → anomalous
    Normalise this to 0–1 so Ashan can threshold it cleanly.
    """
    global _model

    if _model is None:
        log.warning("Anomaly model not loaded — returning 0.0.")
        return 1.0

    # TODO (Ginura): implement this
    # Steps:
    #   score = _model.decision_function(vector.reshape(1, -1))[0]
    #   The raw score range varies — normalise it.
    #   A simple approach: clip to [-0.5, 0.5] then scale to [0, 1]
    #   normalised = 1.0 - (np.clip(score, -0.5, 0.5) + 0.5)
    #   return float(normalised)

    raise NotImplementedError("Ginura: implement get_anomaly_score() in anomaly_model.py")
