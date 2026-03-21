"""
scripts/mock_runner.py — ALL TEAM MEMBERS
===========================================
Simulates the full system on a laptop with no hardware.
Injects fake vision and audio data so you can test your own
module in the context of the whole pipeline.

Usage:
    uv run python scripts/mock_runner.py --scenario people_talking
    uv run python scripts/mock_runner.py --scenario ghost_booking
    uv run python scripts/mock_runner.py --scenario silent_worker
    uv run python scripts/mock_runner.py --scenario ac_noise

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

import sys
import os

# Ensure project root is in path so 'config', 'fusion', 'cloud' resolve correctly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
        timestamp = datetime.now(timezone.utc).isoformat()
        config.audio_queue.put({
            "vad_fired": scenario["vad_fired"],
            "timestamp": timestamp,
        })
        config.anomaly_queue.put({
            "anomaly_score": scenario["anomaly_score"],
            "timestamp":     timestamp,
        })
        time.sleep(1)


def print_state_loop():
    """Reads from queues and prints the fusion decision."""
    log.info("[MOCK] State printer started.")
    
    latest_vision = False
    latest_audio = False
    latest_anomaly = False
    
    while True:
        while not config.vision_queue.empty():
            v = config.vision_queue.get_nowait()
            latest_vision = v["confidence"] >= config.VISION_THRESHOLD
        while not config.audio_queue.empty():
            a = config.audio_queue.get_nowait()
            latest_audio = a["vad_fired"]
        while not config.anomaly_queue.empty():
            m = config.anomaly_queue.get_nowait()
            latest_anomaly = m["anomaly_score"] >= config.ANOMALY_THRESHOLD

        combined_audio = latest_audio and latest_anomaly
        in_use         = latest_vision or combined_audio
        state = "IN_USE" if in_use else "EMPTY"
        log.info(
            f"Vision={'✓' if latest_vision else '✗'}  "
            f"VAD={'Y' if latest_audio else 'N'}  "
            f"Anomaly={'✓' if latest_anomaly else '✗'}  "
            f"→  {state}"
        )
        time.sleep(2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=SCENARIOS.keys(), default="people_talking")
    parser.add_argument("--real-vision", action="store_true", help="Use real camera/vision logic")
    parser.add_argument("--real-audio",  action="store_true", help="Use real mic/audio logic")
    args = parser.parse_args()

    scenario = SCENARIOS[args.scenario]
    log.info(f"Running scenario: '{args.scenario}' — {scenario['description']}")

    # Override the 10-minute timeout for quick local testing (flips to EMPTY in 5s)
    config.EMPTY_TIMEOUT_SECONDS = 5
    log.info("Overrode config.EMPTY_TIMEOUT_SECONDS to 5 for fast testing.")

    # Import the actual real logic (Ashan's fusion and cloud publisher)
    from fusion.fusion import fusion_loop
    from cloud.cloud_publisher import cloud_publisher

    # Prepare threads based on user flags
    threads = []

    # 1. Vision (Real vs Mock)
    if args.real_vision:
        from vision.camera_loop import camera_loop
        threads.append(threading.Thread(target=camera_loop, name="RealVision", daemon=True))
    else:
        threads.append(threading.Thread(target=fake_vision_producer, args=(scenario,), name="MockVision", daemon=True))

    # 2. Audio (Real vs Mock)
    if args.real_audio:
        from audio.audio_loop import audio_loop
        threads.append(threading.Thread(target=audio_loop, name="RealAudio", daemon=True))
    else:
        threads.append(threading.Thread(target=fake_audio_producer, args=(scenario,), name="MockAudio", daemon=True))

    # 3. Always Real (Fusion and Cloud)
    threads.append(threading.Thread(target=fusion_loop, name="FusionLoop", daemon=True))
    threads.append(threading.Thread(target=cloud_publisher, name="CloudPublisher", daemon=True))
    for t in threads:
        t.start()

    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        log.info("Mock runner stopped.")
