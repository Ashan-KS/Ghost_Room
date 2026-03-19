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
    log.info("Fusion loop started.")

    current_state      = EMPTY
    last_signal_time   = 0.0

    while True:
        now = time.time()

        # Drain both queues — take the latest reading from each
        vision_msg = _drain_queue(config.vision_queue)
        audio_msg  = _drain_queue(config.audio_queue)

        # ── Vision gate ───────────────────────────────────────────────────────
        vision_signal = False
        if vision_msg is not None:
            vision_signal = vision_msg["confidence"] >= config.VISION_THRESHOLD
            log.debug(f"Vision conf={vision_msg['confidence']:.2f} → {vision_signal}")

        # ── Audio gate (AND: VAD must fire AND anomaly must be above threshold) ─
        audio_signal = False
        if audio_msg is not None:
            vad_ok   = audio_msg["vad_fired"]
            anom_ok  = audio_msg["anomaly_score"] >= config.ANOMALY_THRESHOLD
            audio_signal = vad_ok and anom_ok
            log.debug(f"Audio vad={vad_ok} anom={audio_msg['anomaly_score']:.2f} → {audio_signal}")

        # ── OR fusion ─────────────────────────────────────────────────────────
        any_signal = vision_signal or audio_signal
        if any_signal:
            last_signal_time = now

        # ── State machine ─────────────────────────────────────────────────────
        time_since_signal = now - last_signal_time if last_signal_time > 0 else float("inf")
        new_state = IN_USE if time_since_signal < config.EMPTY_TIMEOUT_SECONDS else EMPTY

        if new_state != current_state:
            log.info(f"State change: {current_state} → {new_state}")
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
