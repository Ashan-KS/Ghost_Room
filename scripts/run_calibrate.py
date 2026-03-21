import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import sys
import os

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app_config
from anomaly import anomaly_model

def record_baseline():
    fs = app_config.AUDIO_SAMPLE_RATE
    seconds = app_config.CALIBRATION_DURATION_MINUTES * 60
    print(f"Recording {app_config.CALIBRATION_DURATION_MINUTES} minutes of audio for calibration...")
    audio = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
    sd.wait()
    audio = audio.flatten()
    write("calibration.wav", fs, (audio * 32767).astype(np.int16))
    print("Calibration recording saved as calibration.wav")
    return audio

def main():
    audio = record_baseline()
    # Break into chunks and extract features
    chunk_len = app_config.AUDIO_CHUNK_MS * app_config.AUDIO_SAMPLE_RATE
    chunks = [
        audio[i:i + chunk_len]
        for i in range(0, len(audio), chunk_len)
        if len(audio[i:i + chunk_len]) == chunk_len
    ]
    X = np.array([anomaly_model._extract_features(c) for c in chunks])
    print(f"Extracted {len(X)} feature vectors of shape {X.shape[1:]}")
    anomaly_model.train_and_save(X)
    print(f"Model calibrated and saved to {app_config.ANOMALY_MODEL_PATH}")

if __name__ == "__main__":
    main()