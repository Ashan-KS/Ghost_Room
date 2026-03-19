"""
scripts/mock_runner.py — ALL TEAM MEMBERS
===========================================
Simulates the full system on a laptop with no hardware.
Injects fake vision and audio data so you can test your own
module in the context of the whole pipeline.

Usage:
    python scripts/mock_runner.py --scenario people_talking
    python scripts/mock_runner.py --scenario ghost_booking
    python scripts/mock_runner.py --scenario silent_worker
    python scripts/mock_runner.py --scenario ac_noise

Press Ctrl+C to stop.
"""

import threading
import time
import logging
import argparse
from datetime import datetime, timezone

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(threadName)s] %(levelname)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

import config

# ── Scenario definitions ──────────────────────────────────────────────────────
SCENARIOS = {
    "people_talking": {
        "description": "People present, talking — should be IN_USE",
        "vision_confidence": 0.88,
        "vad_fired":         True,
        "anomaly_score":     0.82,
    },
    "ghost_booking": {
        "description": "Room booked but empty — should be EMPTY",
        "vision_confidence": 0.04,
        "vad_fired":         False,
        "anomaly_score":     0.06,
    },
    "silent_worker": {
        "description": "Person present but silent — should be IN_USE (vision only)",
        "vision_confidence": 0.79,
        "vad_fired":         False,
        "anomaly_score":     0.09,
    },
    "ac_noise": {
        "description": "AC unit creates audio — should be EMPTY (anomaly suppresses VAD)",
        "vision_confidence": 0.03,
        "vad_fired":         True,
        "anomaly_score":     0.12,
    },
}


def fake_vision_producer(scenario: dict):
    """Mimics Sachith's camera_loop."""
    log.info(f"[MOCK] Vision producing confidence={scenario['vision_confidence']}")
    while True:
        config.vision_queue.put({
            "confidence": scenario["vision_confidence"],
            "timestamp":  datetime.now(timezone.utc).isoformat(),
        })
        time.sleep(2)


def fake_audio_producer(scenario: dict):
    """Mimics Rahul's audio_loop."""
    log.info(f"[MOCK] Audio producing vad={scenario['vad_fired']} anom={scenario['anomaly_score']}")
    while True:
        config.audio_queue.put({
            "vad_fired":     scenario["vad_fired"],
            "anomaly_score": scenario["anomaly_score"],
            "timestamp":     datetime.now(timezone.utc).isoformat(),
        })
        time.sleep(1)


def print_state_loop():
    """Reads from both queues and prints the fusion decision."""
    log.info("[MOCK] State printer started.")
    while True:
        v = a = None
        while not config.vision_queue.empty():
            v = config.vision_queue.get_nowait()
        while not config.audio_queue.empty():
            a = config.audio_queue.get_nowait()

        if v and a:
            vision_signal = v["confidence"] >= config.VISION_THRESHOLD
            audio_signal  = a["vad_fired"] and a["anomaly_score"] >= config.ANOMALY_THRESHOLD
            in_use        = vision_signal or audio_signal

            state = "IN_USE" if in_use else "EMPTY"
            log.info(
                f"Vision={v['confidence']:.2f}({'✓' if vision_signal else '✗'})  "
                f"VAD={'Y' if a['vad_fired'] else 'N'}  "
                f"Anomaly={a['anomaly_score']:.2f}({'✓' if audio_signal else '✗'})  "
                f"→  {state}"
            )
        time.sleep(2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=SCENARIOS.keys(), default="people_talking")
    args = parser.parse_args()

    scenario = SCENARIOS[args.scenario]
    log.info(f"Running scenario: '{args.scenario}' — {scenario['description']}")

    threads = [
        threading.Thread(target=fake_vision_producer, args=(scenario,), name="MockVision",  daemon=True),
        threading.Thread(target=fake_audio_producer,  args=(scenario,), name="MockAudio",   daemon=True),
        threading.Thread(target=print_state_loop,                        name="StatePrinter",daemon=True),
    ]
    for t in threads:
        t.start()

    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        log.info("Mock runner stopped.")
