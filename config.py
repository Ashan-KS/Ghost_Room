"""
config.py — shared by ALL modules. Import this, never instantiate your own queues.

Queue message contracts (read carefully before writing to any queue):

vision_queue message:
    {
        "confidence": float,    # 0.0–1.0, person detection confidence from MobileNet
        "timestamp":  str,      # ISO format e.g. "2025-03-19T10:15:03"
    }

audio_queue message:
    {
        "vad_fired":     bool,   # True if WebRTC VAD detected speech this chunk
        "timestamp":     str,    # ISO format
    }

anomaly_queue message:
    {
        "anomaly_score": float,  # 0.0–1.0, normalised score from IsolationForest
        "timestamp":     str,    # ISO format
    }
"""

import queue
import os

# ── Shared queues ─────────────────────────────────────────────────────────────
vision_queue  = queue.Queue()   # Sachith  → Ashan
audio_queue   = queue.Queue()   # Rahul    → Ashan
anomaly_queue = queue.Queue()   # Ginura   → Ashan

# ── Decision thresholds ───────────────────────────────────────────────────────
VISION_THRESHOLD  = 0.50   # MobileNet confidence above this → person detected
VAD_THRESHOLD     = 0.60   # VAD confidence above this → speech detected
ANOMALY_THRESHOLD = 0.50   # Anomaly score above this → not baseline (real signal)

# ── State machine ─────────────────────────────────────────────────────────────
EMPTY_TIMEOUT_SECONDS = 300   # 10 minutes of no signal → flip to EMPTY

# ── Hardware flag ─────────────────────────────────────────────────────────────
# Set False on laptops, True when running on Raspberry Pi
USE_PI_HARDWARE = False

# ── Camera settings ───────────────────────────────────────────────────────────
CAMERA_INDEX      = 0       # laptop webcam; ignored when USE_PI_HARDWARE=True
CAMERA_FPS        = 30    # frames per second for inference loop
FRAME_SIZE        = (300, 300)

# ── Audio settings ────────────────────────────────────────────────────────────
AUDIO_SAMPLE_RATE = 16000   # Hz — required by webrtcvad
AUDIO_CHUNK_MS    = 30      # ms per VAD frame (10, 20, or 30 only)
VAD_MODE          = 2       # 0=least aggressive, 3=most aggressive
AUDIO_DEVICE_INDEX = None   # None = system default; set to USB mic index on Pi
CALIBRATION_DURATION_S = 60    # seconds of empty-room recording for calibration
AUDIO_LOG_STATS_INTERVAL_S = 5    # Frequency of audio loop stats logging


# ── GPIO pin assignments (Pi only) ────────────────────────────────────────────
GPIO_RED_LED   = 17   # "Do Not Disturb" — room IN_USE
GPIO_GREEN_LED = 27   # "Available"      — room EMPTY

# ── Cloud / MQTT ──────────────────────────────────────────────────────────────
MQTT_BROKER_HOST  = "localhsot"
MQTT_BROKER_PORT  = 1883
MQTT_TOPIC_STATUS = "room/A/status"     # Pi publishes here
MQTT_TOPIC_CMD    = "room/A/command"    # Pi subscribes here (for maintenance lock)
MQTT_TOPIC_CALIB_PROGRESS = "room/A/calibration/progress"  # Pi publishes calibration progress here
ROOM_ID           = "A"

# ── AWS S3 ────────────────────────────────────────────────────────────────────
ENABLE_S3_LOGGING = False
S3_BUCKET_NAME    = "workspace-agent-logs"
S3_LOG_PREFIX     = "events/"

# ── Model backend ─────────────────────────────────────────────────────────────
# Options:  "mobilenet"  |  "yolo"
MODEL_BACKEND = "yolo"

# ── Model & label file paths ──────────────────────────────────────────────────
# Drop your weights into the models/ folder and update the names below.
#
# MobileNet SSD (TFLite):
#   models/ssd_mobilenet_v1_coco_quant.tflite   ← quantized uint8 (recommended for Pi)
#   models/coco_labels.txt
#
# YOLOv8 (choose ONE format):
#   models/yolov8n.pt        ← Ultralytics native (dev / laptop)
#   models/yolov8n.onnx      ← ONNX export (Pi deployment, needs onnxruntime)
#
_MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
ANOMALY_MODEL_PATH = os.path.join(_MODELS_DIR, "baseline.pkl")

if MODEL_BACKEND == "mobilenet":
    MODEL_PATH = os.path.join(_MODELS_DIR, "ssd_mobilenet_v1_coco_quant.tflite")
    LABEL_PATH = os.path.join(_MODELS_DIR, "coco_labels.txt")
elif MODEL_BACKEND == "yolo":
    MODEL_PATH = os.path.join(_MODELS_DIR, "yolov8n.pt")    # swap to .onnx on Pi
    LABEL_PATH = os.path.join(_MODELS_DIR, "coco_labels.txt")
else:
    raise ValueError(f"Unknown MODEL_BACKEND: {MODEL_BACKEND!r}. Choose 'mobilenet' or 'yolo'.")


