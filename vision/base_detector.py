"""
vision/base_detector.py
========================
Abstract base class that every detector backend must implement.

Swapping models (MobileNet SSD, YOLOv8, …) = swap the concrete class.
The rest of the codebase (camera_loop, fusion, …) only ever calls the
two methods defined here and never imports a backend directly.

Concrete implementations live in:
    vision/mobilenet_detector.py   ← TFLite quantized MobileNet SSD
    vision/yolo_detector.py        ← YOLOv8 via ultralytics or ONNX Runtime
"""

from abc import ABC, abstractmethod
from typing import List
import numpy as np


class BaseDetector(ABC):
    """
    Minimal interface every person-detection backend must satisfy.

    Lifecycle
    ---------
    detector = ConcreteDetector(config)
    detector.load()                    # called once at startup
    conf = detector.run_inference(frame)   # called per frame
    """

    def __init__(self, cfg):
        """
        Args:
            cfg: the imported config module (or any object with the
                 same attributes).  Passed in so each backend can read
                 its own path / threshold / size settings without a
                 hard import of config at module level.
        """
        self.cfg = cfg
        self._loaded = False

    # ------------------------------------------------------------------ #
    #  Abstract interface — every backend MUST implement these two        #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def load(self) -> None:
        """
        Load model weights into memory and prepare for inference.
        Must set self._loaded = True on success.
        Raise RuntimeError if the model file is missing or corrupt.
        """

    @abstractmethod
    def run_inference(self, frame: np.ndarray) -> float:
        """
        Detect persons in a single BGR frame.

        Args:
            frame: np.ndarray shape (H, W, 3), dtype uint8, BGR channel order
                   (exactly what cv2.VideoCapture returns).

        Returns:
            float in [0.0, 1.0].
            • 0.0  — no person detected above cfg.VISION_THRESHOLD
            • >0.0 — highest person confidence score found in the frame
        """

    # ------------------------------------------------------------------ #
    #  Shared helpers available to all concrete subclasses               #
    # ------------------------------------------------------------------ #

    def _ensure_loaded(self) -> None:
        """Call at the top of run_inference to auto-load on first use."""
        if not self._loaded:
            self.load()

    def _load_labels(self, path: str) -> List[str]:
        """
        Read a plain-text label file (one class per line).
        Strips the dummy '???' first line that some COCO label files include.

        Returns:
            list of class-name strings, index-aligned with model outputs.
        """
        with open(path, "r") as fh:
            labels = [line.strip() for line in fh.readlines()]
        if labels and labels[0] == "???":
            labels = labels[1:]
        return labels

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(loaded={self._loaded})"