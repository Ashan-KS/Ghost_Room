"""
fusion/fusion.py — ASHAN
==========================
Reads from vision_queue and audio_queue, applies decision logic,
manages the 10-minute EMPTY timeout, and emits room state changes.

Logic:
  audio_signal = vad_fired AND anomaly_score >= ANOMALY_THRESHOLD
  in_use       = vision_signal OR audio_signal
  EMPTY only declared after EMPTY_TIMEOUT_SECONDS of no signal

State changes are pushed to cloud_publisher via a state_queue.
"""

import logging
import time
from datetime import datetime, timezone

import config
from cloud.cloud_publisher import publish_state

log = logging.getLogger(__name__)

# Room states
IN_USE = "IN_USE"
EMPTY  = "EMPTY"


def fusion_loop():
    """Main fusion thread — runs forever."""
    log.info("Fusion loop started. Waiting for sensors and cloud to connect...")
    time.sleep(2)  # Give MQTT client time to fully connect to EC2

    current_state      = None
    last_signal_time   = 0.0

    # Track latest signals for fusion of independent queues
    latest_vision  = False
    latest_audio   = False
    latest_anomaly = False

    while True:
        now = time.time()

        # Drain all queues — take the latest reading from each
        vision_msg  = _drain_queue(config.vision_queue)
        audio_msg   = _drain_queue(config.audio_queue)
        anomaly_msg = _drain_queue(config.anomaly_queue)

        # ── Vision gate ───────────────────────────────────────────────────────
        if vision_msg is not None:
            latest_vision = vision_msg["confidence"] >= config.VISION_THRESHOLD
            log.debug(f"Vision conf={vision_msg['confidence']:.2f} → {latest_vision}")

        # ── Audio gate (VAD) ──────────────────────────────────────────────────
        if audio_msg is not None:
            latest_audio = audio_msg["vad_fired"]
            log.debug(f"Audio vad={latest_audio}")

        # ── Anomaly gate ──────────────────────────────────────────────────────
        if anomaly_msg is not None:
            latest_anomaly = anomaly_msg["anomaly_score"] >= config.ANOMALY_THRESHOLD

        # ── Fusion ────────────────────────────────────────────────────────────
        # audio and anomaly are combined, then OR'd with vision
        combined_audio = latest_audio and latest_anomaly
        any_signal     = latest_vision or combined_audio

        if any_signal:
            last_signal_time = now

        # ── State machine ─────────────────────────────────────────────────────
        time_since_signal = now - last_signal_time if last_signal_time > 0 else float("inf")
        new_state = IN_USE if time_since_signal < config.EMPTY_TIMEOUT_SECONDS else EMPTY

        if new_state != current_state:
            log.info(
                f"STATE CHANGE: {current_state} → {new_state}  "
                f"[vision={'✓' if latest_vision else '✗'}  "
                f"vad={'✓' if latest_audio else '✗'}  "
                f"anomaly={'✓' if latest_anomaly else '✗'}]"
            )
            current_state = new_state
            publish_state(current_state)

        time.sleep(0.5)   # poll every 500ms


def _drain_queue(q):
    """
    Drain a queue and return only the most recent item.
    Discards any backlog — we only care about the latest reading.
    """
    item = None
    while not q.empty():
        try:
            item = q.get_nowait()
        except Exception:
            break
    return item
