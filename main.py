"""
main.py — entry point for the Raspberry Pi agent.

Run with:  python main.py
On boot:   managed by systemd (see scripts/workspace-agent.service)

Startup sequence:
  1. Check if a saved anomaly baseline exists
     - Yes → load it and skip calibration
     - No  → run 5-min calibration, save baseline.pkl
  2. Spawn four daemon threads: camera, audio, fusion, cloud publisher
  3. Keep main thread alive
"""

import threading
import logging
import os
import sys

from app_config import ANOMALY_MODEL_PATH

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(threadName)s] %(levelname)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def main():
    log.info("Workspace Agent starting up...")

    # ── Step 1: Calibration ───────────────────────────────────────────────────
    from anomaly.calibration import run_calibration
    from anomaly.anomaly_model import load_model

    if os.path.exists(ANOMALY_MODEL_PATH):
        log.info(f"Baseline model found at {ANOMALY_MODEL_PATH} — skipping calibration.")
        load_model()
    else:
        log.info("No baseline found. Starting calibration (do not enter the room)...")
        run_calibration()
        log.info("Calibration complete. Baseline saved.")

    # ── Step 2: Spawn threads ─────────────────────────────────────────────────
    from vision.camera_loop    import camera_loop
    from audio.audio_loop      import audio_loop
    from fusion.fusion         import fusion_loop
    from cloud.cloud_publisher import cloud_publisher

    threads = [
        threading.Thread(target=camera_loop,    name="CameraLoop",    daemon=True),
        threading.Thread(target=audio_loop,     name="AudioLoop",     daemon=True),
        threading.Thread(target=fusion_loop,    name="FusionLoop",    daemon=True),
        threading.Thread(target=cloud_publisher,name="CloudPublisher",daemon=True),
    ]

    for t in threads:
        t.start()
        log.info(f"Started thread: {t.name}")

    log.info("All threads running. Agent is live.")

    # ── Step 3: Keep main thread alive ────────────────────────────────────────
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        log.info("Shutdown requested. Exiting.")
        sys.exit(0)


if __name__ == "__main__":
    main()
