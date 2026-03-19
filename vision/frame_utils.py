"""
vision/frame_utils.py — SACHITH
=================================
Camera capture and frame preprocessing utilities.

Hardware abstraction lives here — swap USE_PI_HARDWARE flag in config.py,
everything else stays identical.
"""

import logging
import numpy as np
import cv2
import config

log = logging.getLogger(__name__)

_cap = None   # OpenCV VideoCapture, reused across calls


def get_camera_frame():
    """
    Capture a single frame from the camera.

    Returns:
        numpy array (H, W, 3) BGR, or None on failure.
    """
    global _cap

    if config.USE_PI_HARDWARE:
        # TODO (Sachith): implement Pi Camera capture using picamera2
        # from picamera2 import Picamera2
        # ...
        raise NotImplementedError("Sachith: implement Pi Camera capture in frame_utils.py")
    else:
        # Laptop webcam
        if _cap is None or not _cap.isOpened():
            _cap = cv2.VideoCapture(config.CAMERA_INDEX)
            if not _cap.isOpened():
                log.error("Could not open webcam.")
                return None

        ret, frame = _cap.read()
        return frame if ret else None


def preprocess_frame(frame) -> np.ndarray:
    """
    Preprocess a raw frame for MobileNet SSD input.

    Args:
        frame: numpy array (H, W, 3) BGR

    Returns:
        numpy array (1, 300, 300, 3) uint8, RGB channel order
    """
    resized = cv2.resize(frame, config.FRAME_SIZE)
    rgb     = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return np.expand_dims(rgb, axis=0).astype(np.uint8)


def release_camera():
    """Call this on shutdown to release the camera resource."""
    global _cap
    if _cap is not None:
        _cap.release()
        _cap = None
