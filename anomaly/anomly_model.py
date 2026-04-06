import numpy as np
import joblib
import os
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import config
import librosa

_model = None
_scaler = None
_stat_mean = None
_stat_std = None
_stat_threshold = None
_iso_threshold = None
_model_mtime = 0

# ML Fine-tuning parameters
N_MFCC = 13
N_FFT_MAX = 512
ANOMALY_CONTAMINATION = 0.05
ANOMALY_RANDOM_STATE = 42
ANOMALY_PERCENTILE = 5


def extract_anomaly_features(audio: np.ndarray, sr: int = config.AUDIO_SAMPLE_RATE) -> np.ndarray:
    """
    Extract features from audio vector: 13 MFCC means, 13 MFCC stds, energy, ZCR.
    Returns vector of shape (28,).
    """
    if len(audio.shape) > 1:
        audio = audio.flatten()
    n_fft = min(N_FFT_MAX, len(audio))
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC, n_fft=n_fft)
    mfcc_means = np.mean(mfcc, axis=1)
    mfcc_stds  = np.std(mfcc, axis=1)
    energy = np.mean(audio ** 2)
    zcr = np.mean(librosa.feature.zero_crossing_rate(audio, frame_length=n_fft))
    features = np.hstack([mfcc_means, mfcc_stds, energy, zcr])
    return features.astype(np.float32)

def train_and_save(X: np.ndarray):
    global _model, _scaler, _stat_mean, _stat_std, _stat_threshold, _iso_threshold
    # Statistical baseline
    _stat_mean = np.mean(X, axis=0)
    _stat_std = np.std(X, axis=0)
    stat_scores = np.mean(np.abs((X - _stat_mean) / (_stat_std + 1e-6)), axis=1)
    _stat_threshold = np.mean(stat_scores) + 3 * np.std(stat_scores)

    # Scaler for IsolationForest compatibility
    _scaler = StandardScaler()
    X_scaled = _scaler.fit_transform(X)

    # IsolationForest
    _model = IsolationForest(contamination=ANOMALY_CONTAMINATION, random_state=ANOMALY_RANDOM_STATE)
    _model.fit(X_scaled)
    scores = _model.decision_function(X_scaled)
    _iso_threshold = np.percentile(scores, ANOMALY_PERCENTILE)

    # Save everything needed for inference
    save_data = {
        "isoforest": _model,
        "scaler": _scaler,
        "stat_mean": _stat_mean,
        "stat_std": _stat_std,
        "stat_threshold": _stat_threshold,
        "iso_threshold": _iso_threshold
    }
    os.makedirs(os.path.dirname(config.ANOMALY_MODEL_PATH), exist_ok=True)
    joblib.dump(save_data, config.ANOMALY_MODEL_PATH)

def load_model():
    global _model, _scaler, _stat_mean, _stat_std, _stat_threshold, _iso_threshold, _model_mtime
    data = joblib.load(config.ANOMALY_MODEL_PATH)
    _model        = data["isoforest"]
    _scaler       = data["scaler"]
    _stat_mean    = data["stat_mean"]
    _stat_std     = data["stat_std"]
    _stat_threshold = data["stat_threshold"]
    _iso_threshold  = data["iso_threshold"]
    if os.path.exists(config.ANOMALY_MODEL_PATH):
        _model_mtime = os.path.getmtime(config.ANOMALY_MODEL_PATH)

def get_anomaly_score(vec: np.ndarray) -> float:
    global _model, _scaler, _stat_mean, _stat_std, _stat_threshold, _iso_threshold, _model_mtime
    
    # Hot-reload if the file was modified
    if os.path.exists(config.ANOMALY_MODEL_PATH):
        current_mtime = os.path.getmtime(config.ANOMALY_MODEL_PATH)
        if current_mtime > _model_mtime:
            # Re-load if a new calibration file is detected
            load_model()
            
    # Ensure model loaded
    if _model is None or _scaler is None:
        load_model()
    # Statistical score
    stat_score = np.mean(np.abs((vec - _stat_mean) / (_stat_std + 1e-6)))
    # IF score (High anomaly is LOWER value!)
    iso_score = _model.decision_function(_scaler.transform([vec]))[0]
    # 1 = anomaly if either triggers
    is_anomaly = (stat_score > _stat_threshold) or (iso_score < _iso_threshold)
    return 1 if is_anomaly else 0