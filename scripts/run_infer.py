import sounddevice as sd
import numpy as np
import queue
import threading
import time
from datetime import datetime
from anomly import anomly_model
import config

def listen_and_detect(anomaly_queue: queue.Queue, stop_event: threading.Event):
    fs = config.AUDIO_SAMPLE_RATE
    chunk_len = config.AUDIO_CHUNK_MS * fs
    print("Real-time anomaly detection started. (CTRL+C to stop)")

    anomly_model.load_model()

    try:
        while not stop_event.is_set():
            audio = sd.rec(int(chunk_len), samplerate=fs, channels=1)
            sd.wait()
            audio = audio.flatten()

            # Optional normalization
            if np.max(np.abs(audio)) > 0:
                audio = audio / np.max(np.abs(audio))

            feat = anomly_model._extract_features(audio)
            anomaly_score = anomly_model.get_anomaly_score(feat)
            ts = datetime.now().isoformat(timespec="seconds")

            anomaly_queue.put({
                'anomaly_score': anomaly_score,
                'timestamp': ts,
            })
    except KeyboardInterrupt:
        print("Stopped by user.")
        stop_event.set()

def main():
    q = queue.Queue()
    stop_event = threading.Event()

    listener_thread = threading.Thread(target=listen_and_detect, args=(q, stop_event))
    input("Press Enter to start real-time detection... ")
    listener_thread.start()

    try:
        while not stop_event.is_set():
            try:
                msg = q.get(timeout=1)
                status = "ANOMALY" if msg['anomaly_score'] else "normal"
                print(f"[{msg['timestamp']}]  {status}")
                # Here you could pipe this to other integrations
            except queue.Empty:
                continue
    except KeyboardInterrupt:
        print("Stopping detection ...")
        stop_event.set()
    listener_thread.join()

if __name__ == "__main__":
    main()