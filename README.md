# Workspace Agent — Edge AI Room Monitor

Autonomous meeting room occupancy detection using a Raspberry Pi 4,
camera, microphone, and AWS cloud integration.

---

## Team & ownership

| Module | Owner | Files |
|---|---|---|
| Vision (MobileNet SSD) | Sachith | `vision/` |
| Audio (VAD + features) | Rahul   | `audio/`  |
| Anomaly detection      | Ginura  | `anomaly/`|
| Fusion + cloud + GPIO  | Ashan   | `fusion/`, `cloud/`, `gpio/`, `main.py` |

---

## Setup (everyone does this first)

```bash
git clone <repo-url>
cd workspace-agent
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Download MobileNet weights (see `models/README.txt`).

---

## Running your module in isolation

Each person tests their own module independently using the mock runner:

```bash
# Test with a fake scenario — no hardware needed
python scripts/mock_runner.py --scenario people_talking
python scripts/mock_runner.py --scenario ghost_booking
python scripts/mock_runner.py --scenario silent_worker
python scripts/mock_runner.py --scenario ac_noise
```

---

## Running tests

```bash
pytest tests/test_vision.py  -v   # Sachith
pytest tests/test_audio.py   -v   # Rahul
pytest tests/test_anomaly.py -v   # Ginura
pytest tests/test_fusion.py  -v   # Ashan
pytest                            # run all
```

All tests must pass before pushing to main.

---

## Running the full system (laptop)

```bash
# config.py: USE_PI_HARDWARE = False
python main.py
```

## Running on Raspberry Pi

```bash
# config.py: USE_PI_HARDWARE = True, set MQTT_BROKER_HOST
python main.py

# Or install as a systemd service (auto-starts on boot):
sudo cp scripts/workspace-agent.service /etc/systemd/system/
sudo systemctl enable workspace-agent
sudo systemctl start workspace-agent
sudo journalctl -u workspace-agent -f   # view logs
```

---

## Interface contracts — READ BEFORE CODING

These are the agreed message shapes for each queue.
**Do not change these without telling the whole team.**

### `vision_queue` (Sachith → Ashan)
```python
{
    "confidence": float,   # 0.0–1.0, MobileNet person confidence
    "timestamp":  str,     # ISO format "2025-03-19T10:15:03+00:00"
}
```

### `audio_queue` (Rahul → Ashan)
```python
{
    "vad_fired":     bool,   # True if WebRTC VAD detected speech
    "anomaly_score": float,  # 0.0–1.0, normalised IsolationForest score
    "timestamp":     str,    # ISO format
}
```

### `feature_queue` (Rahul → Ginura)
```python
numpy.ndarray, shape (14,), dtype float32
# [rms, mfcc_0, mfcc_1, ..., mfcc_12]
```

---

## Decision logic

```
audio_signal = vad_fired AND anomaly_score >= 0.50
vision_signal = confidence >= 0.50
in_use = vision_signal OR audio_signal
EMPTY declared after 10 minutes of no signal
```

---

## Project structure

```
workspace-agent/
├── main.py                  # Entry point — Ashan
├── config.py                # Shared config, queues, thresholds — everyone imports this
├── requirements.txt
├── vision/
│   ├── camera_loop.py       # Thread: captures frames, pushes to vision_queue — Sachith
│   ├── vision_inference.py  # MobileNet SSD inference — Sachith
│   └── frame_utils.py       # Camera capture + preprocessing — Sachith
├── audio/
│   ├── audio_loop.py        # Thread: captures audio, pushes to queues — Rahul
│   ├── vad_processor.py     # WebRTC VAD wrapper — Rahul
│   └── feature_extractor.py # Extracts (14,) feature vector — Rahul ← Ginura depends on this
├── anomaly/
│   ├── calibration.py       # 5-min calibration, trains baseline — Ginura
│   └── anomaly_model.py     # IsolationForest train/load/infer — Ginura
├── fusion/
│   └── fusion.py            # OR gate + state machine + 10-min timer — Ashan
├── cloud/
│   ├── cloud_publisher.py   # MQTT publish/subscribe — Ashan
│   └── dashboard/
│       └── app.py           # Streamlit dashboard on EC2 — Ashan
├── gpio/
│   └── actuator.py          # LED control via GPIO — Ashan
├── models/
│   ├── README.txt           # How to download MobileNet weights
│   └── ssd_mobilenet_v2_coco_quant.tflite   # download separately
├── tests/
│   ├── test_vision.py       # Sachith's tests
│   ├── test_audio.py        # Rahul's tests
│   ├── test_anomaly.py      # Ginura's tests
│   └── test_fusion.py       # Ashan's tests
└── scripts/
    ├── mock_runner.py        # Simulate full pipeline on laptop — everyone
    └── workspace-agent.service  # systemd service for Pi autostart
```
