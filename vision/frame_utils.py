"""
vision/frame_utils.py
======================
Camera capture and frame preprocessing utilities.

Hardware abstraction lives here — swap USE_PI_HARDWARE in config.py
and everything else stays identical.

Preprocessing note
------------------
preprocess_frame() is kept here for backward compatibility and used by
MobileNetDetector._preprocess().  YoloDetector has its own preprocessing
since YOLO expects a different tensor format (NCHW float32 0-1).
"""

import logging
from typing import Optional
import numpy as np
import cv2
import config

log = logging.getLogger(__name__)

_cap          = None   # OpenCV VideoCapture, reused across calls
_pi_camera    = None   # picamera2 instance, reused across calls


# ── Public API ────────────────────────────────────────────────────────────────

def get_camera_frame() -> Optional[np.ndarray]:
    """
    Capture a single frame from the configured camera source.

    Returns:
        numpy array (H, W, 3) dtype uint8, BGR channel order — same as
        cv2.VideoCapture.read() — or None on failure.
    """
    # if config.USE_PI_HARDWARE:
    #     return _get_pi_frame()
    # else:
    return _get_webcam_frame()


def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """
    Preprocess a raw BGR frame for MobileNet SSD input.

    Args:
        frame: numpy array (H, W, 3) BGR

    Returns:
        numpy array (1, 300, 300, 3) uint8, RGB channel order.
        Height and width come from config.FRAME_SIZE.
    """
    w, h    = config.FRAME_SIZE
    resized = cv2.resize(frame, (w, h))
    rgb     = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return np.expand_dims(rgb, axis=0).astype(np.uint8)


def release_camera() -> None:
    """Release camera resources. Call this on clean shutdown."""
    global _cap, _pi_camera

    if _cap is not None:
        _cap.release()
        _cap = None
        log.info("Webcam released.")

    if _pi_camera is not None:
        try:
            _pi_camera.stop()
        except Exception:
            pass
        _pi_camera = None
        log.info("Pi Camera released.")


# ── Private helpers ────────────────────────────────────────────────────────────

def _get_webcam_frame() -> Optional[np.ndarray]:
    """Capture from USB/laptop webcam using OpenCV."""
    global _cap

    if _cap is None or not _cap.isOpened():
        _cap = cv2.VideoCapture(config.CAMERA_INDEX)
        if not _cap.isOpened():
            log.error(
                f"Could not open webcam at index {config.CAMERA_INDEX}. "
                "Try changing CAMERA_INDEX in config.py."
            )
            return None
        log.info(f"Webcam opened (index={config.CAMERA_INDEX}).")

    ret, frame = _cap.read()
    if not ret:
        log.warning("Webcam read failed — releasing and retrying next cycle.")
        _cap.release()
        _cap = None
        return None

    return frame


# def _get_pi_frame() -> Optional[np.ndarray]:
#     """
#     Capture from the Raspberry Pi Camera Module using picamera2.

#     picamera2 returns RGB arrays; we convert to BGR to keep the rest of
#     the pipeline consistent with OpenCV convention.

#     Requires:
#         sudo apt install python3-picamera2
#     """
#     global _pi_camera

#     if _pi_camera is None:
#         try:
#             from picamera2 import Picamera2  # noqa: PLC0415
#             _pi_camera = Picamera2()
#             # Still config gives a full-res JPEG-ready frame; video config is
#             # lower-latency for continuous capture at CAMERA_FPS.
#             cfg = _pi_camera.create_video_configuration(
#                 main={"size": (640, 480), "format": "RGB888"}
#             )
#             _pi_camera.configure(cfg)
#             _pi_camera.start()
#             log.info("Pi Camera started via picamera2.")
#         except ImportError:
#             log.error(
#                 "picamera2 not installed. Run: sudo apt install python3-picamera2"
#             )
#             return None
#         except Exception as exc:
#             log.error(f"Pi Camera init failed: {exc}")
#             _pi_camera = None
#             return None

#     try:
#         # capture_array() returns an RGB888 numpy array (H, W, 3)
#         rgb_frame = _pi_camera.capture_array()
#         # Convert RGB → BGR for OpenCV/detector consistency
#         return cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
#     except Exception as exc:
#         log.error(f"Pi Camera capture failed: {exc}")
#         return None