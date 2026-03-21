"""
tests/test_anomaly.py — GINURA
================================
Run with:  python -m pytest tests/test_anomaly.py -v

Tests you must pass before pushing:
  1. train_and_save() creates a .pkl file
  2. load_model() loads without error
  3. get_anomaly_score() returns float in [0, 1]
  4. Baseline data scores lower than noisy data (sanity check)
"""

import numpy as np
import os
import pytest
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import app_config

TEST_MODEL_PATH = "models/test_baseline.pkl"


def _make_baseline_data(n=200) -> np.ndarray:
    """Simulate quiet-room feature vectors — low RMS, flat MFCCs."""
    rng = np.random.default_rng(42)
    # 28 features by default, but change to match your model's required shape
    return rng.normal(loc=0.01, scale=0.005, size=(n, 28)).astype(np.float32)

def _make_noisy_vector() -> np.ndarray:
    """Simulate a loud, speech-like vector — high RMS, varied MFCCs."""
    v = np.random.uniform(0.5, 1.0, size=(28,)).astype(np.float32)
    return v

@pytest.fixture(autouse=True)
def patch_model_path(monkeypatch):
    """Use a test model path so we don't overwrite the real baseline."""
    monkeypatch.setattr(app_config, "ANOMALY_MODEL_PATH", TEST_MODEL_PATH)
    yield
    if os.path.exists(TEST_MODEL_PATH):
        os.remove(TEST_MODEL_PATH)


def test_train_and_save_creates_file():
    from anomaly.anomaly_model import train_and_save
    X = _make_baseline_data()
    train_and_save(X)
    assert os.path.exists(TEST_MODEL_PATH), "baseline.pkl was not created"


def test_load_model_no_error():
    from anomaly.anomaly_model import train_and_save, load_model
    X = _make_baseline_data()
    train_and_save(X)
    load_model()   # should not raise


def test_get_anomaly_score_range():
    from anomaly.anomaly_model import train_and_save, load_model, get_anomaly_score
    X = _make_baseline_data()
    train_and_save(X)
    load_model()

    baseline_vector = _make_baseline_data(n=1)[0]
    score = get_anomaly_score(baseline_vector)
    assert isinstance(score, (float, int)), f"Expected float or int, got {type(score)}"
    assert 0.0 <= score <= 1.0, f"Score {score} out of range [0, 1]"


def test_noisy_scores_higher_than_baseline():
    """
    Core sanity check: vectors unlike the training data should score higher
    than vectors similar to training data.
    """
    from anomaly.anomaly_model import train_and_save, load_model, get_anomaly_score
    X = _make_baseline_data(n=300)
    train_and_save(X)
    load_model()

    baseline_scores = [get_anomaly_score(_make_baseline_data(1)[0]) for _ in range(20)]
    noisy_scores    = [get_anomaly_score(_make_noisy_vector())       for _ in range(20)]

    avg_baseline = np.mean(baseline_scores)
    avg_noisy    = np.mean(noisy_scores)

    assert avg_noisy > avg_baseline, (
        f"Noisy avg ({avg_noisy:.3f}) should be > baseline avg ({avg_baseline:.3f}). "
        f"Check your score normalisation."
    )