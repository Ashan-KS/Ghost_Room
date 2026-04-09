"""
vision/camera_loop.py
======================
Continuously captures frames at config.CAMERA_FPS and runs inference.
Puts results into config.vision_queue for Ashan's fusion module to consume.

Queue message shape (DO NOT change without telling Ashan):
    {
        "confidence": float,      # 0.0–1.0, highest person score in frame
        "timestamp":  str,        # ISO-8601 UTC e.g. "2025-03-21T10:45:00.123456+00:00"
    }

Nothing in this file needs to change when you swap models or hardware.
All knobs live in config.py.
"""

import os
os.environ["YOLO_VERBOSE"] = "False"   # suppress ultralytics before any import
 
import time
import logging
import cv2
from datetime import datetime, timezone
from typing import List
 
import config
from vision.vision_inference import get_detections
from vision.frame_utils import get_camera_frame, release_camera
 
log = logging.getLogger(__name__)
 
 
# ── Drawing helper (kept here so camera_loop is self-contained) ───────────────
 
def _draw_detections(frame, detections: List[dict]) -> None:
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        label = f"{det['label']} {det['confidence']:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 2)
        cv2.putText(
            frame, label, (x1, max(y1 - 8, 14)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 0), 2,
        )
 
 
# ── Optional downscale for faster pre-processing ──────────────────────────────
 
def _maybe_downscale(frame):
    """
    Downscale frame before inference if config.INFERENCE_SCALE < 1.0.
    Returns (inference_frame, scale_x, scale_y) so boxes can be
    mapped back to the original display frame.
    """
    scale = getattr(config, "INFERENCE_SCALE", 1.0)
    if scale >= 1.0:
        return frame, 1.0, 1.0
 
    h, w = frame.shape[:2]
    new_w, new_h = int(w * scale), int(h * scale)
    small = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return small, 1.0 / scale, 1.0 / scale
 
 
def _scale_boxes(detections: List[dict], sx: float, sy: float) -> List[dict]:
    """Scale bounding boxes back to display-frame coordinates."""
    if sx == 1.0 and sy == 1.0:
        return detections
    scaled = []
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        scaled.append({
            **det,
            "box": (int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy)),
        })
    return scaled
 
 
# ── Main loop ─────────────────────────────────────────────────────────────────
 
def camera_loop() -> None:
    """
    Main camera + display thread.
 
    CPU budget per frame at 5 FPS (~200 ms total):
        capture       ~5–10 ms
        downscale     ~1–2 ms   (if INFERENCE_SCALE < 1.0)
        inference     ~60–150 ms (MobileNet/YOLO on Pi)
        draw + imshow ~2–5 ms
        waitKey       remainder  (blocks, not a spin)
    """
    log.info(
        f"Camera loop started — backend={config.MODEL_BACKEND}, "
        f"fps={config.CAMERA_FPS}, threshold={config.VISION_THRESHOLD}, "
        f"inference_scale={getattr(config, 'INFERENCE_SCALE', 1.0)}"
    )
 
    interval_s  = 1.0 / config.CAMERA_FPS      # seconds
    interval_ms = int(interval_s * 1000)        # milliseconds for waitKey
 
    window_name = "Occupancy Detection — press Q to quit"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
 
    try:
        from main import monitoring_active
        while monitoring_active.is_set():
            t0 = time.monotonic()
 
            # ── 1. Capture ────────────────────────────────────────────────
            frame = get_camera_frame()
            if frame is None:
                log.warning("Camera returned None — skipping frame.")
                time.sleep(interval_s)
                continue
 
            # ── 2. Downscale for inference (CPU saving) ───────────────────
            inf_frame, sx, sy = _maybe_downscale(frame)
 
            # ── 3. Single inference call — derive BOTH detections & confidence
            detections   = get_detections(inf_frame)          # one forward pass
            detections   = _scale_boxes(detections, sx, sy)   # map back to display coords
            person_count = len(detections)
            confidence   = max((d["confidence"] for d in detections), default=0.0)
 
            # ── 4. Draw onto display frame ────────────────────────────────
            _draw_detections(frame, detections)
 
            occ_label = f"Occupancy: {person_count} person{'s' if person_count != 1 else ''}"
            cv2.putText(
                frame, occ_label, (10, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2,
            )
            cv2.putText(
                frame,
                f"model: {config.MODEL_BACKEND}",
                (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1,
            )
 
            cv2.imshow(window_name, frame)
 
            # ── 5. Push to vision_queue ────────────────────────────────────
            message = {
                "confidence":   confidence,
                "person_count": person_count,
                "timestamp":    datetime.now(timezone.utc).isoformat(),
            }
            try:
                config.vision_queue.put_nowait(message)
            except Exception:
                log.warning("vision_queue full — frame dropped (consumer too slow).")
 
            status = "OCCUPIED" if confidence >= config.VISION_THRESHOLD else "vacant "
            log.debug(
                f"[vision_queue] {status} | "
                f"persons={person_count} | "
                f"confidence={confidence:.2f} | "
                f"queue_size={config.vision_queue.qsize()}"
            )
 
            # ── 6. Adaptive waitKey — blocks for remainder of frame budget ─
            # This is the key CPU saving: instead of waitKey(1) spinning
            # every ms, we block for the full remaining budget.
            elapsed_ms  = int((time.monotonic() - t0) * 1000)
            wait_ms     = max(1, interval_ms - elapsed_ms)
            key = cv2.waitKey(wait_ms) & 0xFF
            if key in (ord("q"), 27):
                break
 
    except Exception as exc:
        log.error(f"Camera loop fatal error: {exc}", exc_info=True)
 
    finally:
        release_camera()
        cv2.destroyAllWindows()
        log.info("Camera loop exited — camera released.")