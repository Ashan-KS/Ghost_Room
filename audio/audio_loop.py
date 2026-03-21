"""
audio/audio_loop.py — RAHUL
=============================
Continuously captures audio chunks, runs VAD, extracts features,
and puts results into two queues:
  - audio_queue   → Ashan's fusion module  (vad result + anomaly score)
  - feature_queue → Ginura's anomaly model (raw feature vector)

Your tasks:
  1. Implement the capture loop using PyAudio
  2. Call vad_processor.is_speech() on each chunk
  3. Call feature_extractor.extract() to get the 14-dim feature vector
  4. Put the feature vector into feature_queue for Ginura
  5. Get the anomaly score back from Ginura's model
  6. Put the combined result into audio_queue for Ashan

Do NOT change the audio_queue message shape without telling Ashan.
Do NOT change the feature vector shape without telling Ginura.
"""

import logging
import time
from datetime import datetime, timezone

import config
from audio.vad_processor    import is_speech, get_audio_chunk
from audio.feature_extractor import extract_features
from anomaly.anomaly_model  import get_anomaly_score

log = logging.getLogger(__name__)


def audio_loop():
    """Main audio thread — runs forever."""
    log.info("Audio loop started.")

    while True:
        try:
            chunk = get_audio_chunk()
            if chunk is None:
                time.sleep(0.01)
                continue

            # Run VAD
            vad_fired = is_speech(chunk)

            # Extract features → send to Ginura's model
            feature_vector = extract_features(chunk)
            config.feature_queue.put(feature_vector)

            # Get anomaly score from Ginura's model — handle if not implemented yet
            try:
                anomaly_score = get_anomaly_score(feature_vector)
            except NotImplementedError:
                anomaly_score = 0.0
                log.debug("Anomaly model not implemented — defaulting to 0.0")

            # Put separate messages into respective queues
            timestamp = datetime.now(timezone.utc).isoformat()
            
            config.audio_queue.put({
                "vad_fired": vad_fired,
                "timestamp": timestamp,
            })
            
            config.anomaly_queue.put({
                "anomaly_score": anomaly_score,
                "timestamp":     timestamp,
            })

            log.debug(f"Audio → vad={vad_fired}, anomaly={anomaly_score:.2f}")

        except Exception as e:
            log.error(f"Audio loop error: {e}")
            time.sleep(0.1)
