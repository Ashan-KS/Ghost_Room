"""
audio/feature_extractor.py — RAHUL
=====================================
Extracts a 14-dimensional feature vector from a raw audio chunk.

OUTPUT SHAPE CONTRACT — do not change this without telling Ginura:
    numpy.ndarray, shape (14,), dtype float32
    Index 0:     RMS energy
    Index 1–13:  MFCC coefficients 0–12 (13 values)

Ginura's IsolationForest is trained on exactly this shape.
If you change it, her model breaks. Agree on changes first.

Install: pip install librosa numpy
"""

import logging
import numpy as np
import config

log = logging.getLogger(__name__)

FEATURE_DIM = 1 + config.N_MFCC   # 14 total: 1 RMS + 13 MFCCs


def extract_features(chunk: bytes) -> np.ndarray:
    """
    Extract audio features from a raw PCM chunk.

    Args:
        chunk: bytes — raw int16 PCM at AUDIO_SAMPLE_RATE Hz

    Returns:
        numpy.ndarray shape (14,) float32 — [rms, mfcc_0..mfcc_12]
    """
    try:
        import librosa

        # 1. Convert bytes → numpy float32 samples (normalised to [-1, 1])
        samples = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0

        # 2. RMS energy
        rms = np.sqrt(np.mean(samples ** 2))

        # 3. MFCCs via librosa
        #    n_fft must be <= number of samples (480 for 30ms @ 16kHz)
        n_fft = min(config.N_FFT_MAX, len(samples))
        mfccs = librosa.feature.mfcc(
            y=samples,
            sr=config.AUDIO_SAMPLE_RATE,
            n_mfcc=config.N_MFCC,
            n_fft=n_fft,
        )
        mfcc_mean = np.mean(mfccs, axis=1)   # shape (13,)

        # 4. Concatenate into (14,) vector
        vector = np.concatenate([[rms], mfcc_mean]).astype(np.float32)

        return vector

    except Exception as e:
        log.error(f"Feature extraction error: {e}")
        return get_zero_vector()


def get_zero_vector() -> np.ndarray:
    """
    Returns a zero feature vector of the correct shape.
    Ginura can use this to initialise her model before real data arrives.
    """
    return np.zeros(FEATURE_DIM, dtype=np.float32)
