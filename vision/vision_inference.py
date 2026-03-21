"""
vision/vision_inference.py
===========================
Public API surface for the vision subsystem.

This module keeps the original interface (run_inference, _load_model)
intact so camera_loop.py and any external callers don't change.
Internally it delegates to the backend selected in config.MODEL_BACKEND.

Supported backends (set MODEL_BACKEND in config.py):
    "mobilenet"  →  MobileNetDetector (TFLite quantized SSD)
    "yolo"       →  YoloDetector      (ultralytics or ONNX Runtime)

To add a new backend:
    1. Subclass BaseDetector in vision/your_detector.py
    2. Add an elif branch in _get_detector() below
    3. Change MODEL_BACKEND and MODEL_PATH in config.py
    No other files need to change.
"""

import logging
from typing import Optional
import numpy as np
import config
from vision.base_detector import BaseDetector

log = logging.getLogger(__name__)

# Module-level detector instance — loaded once, reused every frame.
_detector: Optional[BaseDetector] = None


# ── Backend registry ───────────────────────────────────────────────────────────

def _get_detector() -> BaseDetector:
    """
    Instantiate (but don't load) the detector for config.MODEL_BACKEND.
    Add new backends here.
    """
    backend = config.MODEL_BACKEND

    if backend == "mobilenet":
        from vision.mobilenet_detector import MobileNetDetector
        return MobileNetDetector(config)

    elif backend == "yolo":
        from vision.yolo_detector import YoloDetector
        return YoloDetector(config)

    else:
        raise ValueError(
            f"Unknown MODEL_BACKEND: {backend!r}. "
            "Valid options: 'mobilenet', 'yolo'."
        )


# ── Original public API (unchanged contract) ───────────────────────────────────

def _load_model() -> None:
    """
    Load the model backend selected in config.py.
    Called automatically on first run_inference(); can also be called
    explicitly at startup to pay the load cost up-front.
    """
    global _detector
    _detector = _get_detector()
    _detector.load()
    log.info(f"Detector loaded: {_detector}")


def run_inference(frame: np.ndarray) -> float:
    """
    Run person detection on a single frame.

    Args:
        frame: numpy array (H, W, 3) BGR image from OpenCV.

    Returns:
        float in [0.0, 1.0].
        0.0 = no person detected above config.VISION_THRESHOLD.
        >0.0 = highest person confidence found in this frame.
    """
    global _detector

    # Lazy-load on first call
    if _detector is None:
        _load_model()

    try:
        return _detector.run_inference(frame)
    except Exception as exc:
        log.error(f"Inference error: {exc}")
        return 0.0


def get_detections(frame: np.ndarray) -> list[dict]:
    """
    Extended helper — returns all person detections as structured dicts.

    Returns:
        list of {
            "confidence": float,
            "label":      "person",
            "box":        (x1, y1, x2, y2)  pixel coords
        }

    Used by debug/visualisation scripts; NOT consumed by camera_loop.
    """
    global _detector

    if _detector is None:
        _load_model()

    if hasattr(_detector, "get_detections"):
        return _detector.get_detections(frame)

    log.warning(f"{type(_detector).__name__} has no get_detections(); returning [].")
    return []