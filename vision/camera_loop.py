"""
vision/camera_loop.py — SACHITH
================================
Continuously captures frames at CAMERA_FPS and runs MobileNet inference.
Puts results into vision_queue for Ashan's fusion module to consume.

Your tasks:
  1. Implement get_camera_frame() for both laptop and Pi hardware
  2. Call run_inference() from vision_inference.py on each frame
  3. Put the result dict into vision_queue — shape defined in app_config.py

Do NOT change the queue message shape without telling Ashan.
"""

import time
import logging
from datetime import datetime, timezone

import app_config
from vision.vision_inference import run_inference
from vision.frame_utils import get_camera_frame

log = logging.getLogger(__name__)


def camera_loop():
    """Main camera thread — runs forever."""
    log.info("Camera loop started.")
    interval = 1.0 / app_config.CAMERA_FPS   # seconds between captures

    while True:
        loop_start = time.time()

        try:
            frame = get_camera_frame()
            if frame is None:
                log.warning("Camera returned None frame — skipping.")
                time.sleep(interval)
                continue

            confidence = run_inference(frame)

            message = {
                "confidence": confidence,
                "timestamp":  datetime.now(timezone.utc).isoformat(),
            }
            app_config.vision_queue.put(message)

            log.debug(f"Vision → confidence={confidence:.2f}")

        except Exception as e:
            log.error(f"Camera loop error: {e}")

        # sleep the remainder of the interval
        elapsed = time.time() - loop_start
        time.sleep(max(0, interval - elapsed))
