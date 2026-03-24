"""
vision/mobilenet_detector.py
=============================
Concrete detector backend: quantized MobileNet SSD v1 via TFLite.

Model file expected at:  config.MODEL_PATH
Label file expected at:  config.LABEL_PATH

Download weights:
    wget https://storage.googleapis.com/download.tensorflow.org/models/tflite/ \\
         coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip
    unzip coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip
    mv detect.tflite    models/ssd_mobilenet_v1_coco_quant.tflite
    mv labelmap.txt     models/coco_labels.txt

TFLite output tensor order for this model
------------------------------------------
  index 0 → boxes   shape (1, N, 4)   float32  normalised [ymin,xmin,ymax,xmax]
  index 1 → classes shape (1, N)      float32  COCO class ids
  index 2 → scores  shape (1, N)      float32  confidence 0–1
  index 3 → count   shape (1,)        float32  valid detection count

Person class id in COCO = 0  (after stripping the '???' dummy label).
"""

import logging
from typing import List
import numpy as np
import cv2
import tensorflow as tf

from vision.base_detector import BaseDetector

log = logging.getLogger(__name__)

# COCO class id for "person" (0-indexed, after stripping '???' header)
_PERSON_CLASS_ID = 0


class MobileNetDetector(BaseDetector):
    """
    TFLite quantized MobileNet SSD v1 COCO.

    Input tensor : (1, 300, 300, 3) uint8  RGB
    Output tensors: boxes, classes, scores, count  (see module docstring)
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self._interpreter = None
        self._input_details = None
        self._output_details = None
        self._input_h = 300
        self._input_w = 300
        self._labels: List[str] = []

    # ------------------------------------------------------------------ #
    #  BaseDetector interface                                             #
    # ------------------------------------------------------------------ #

    def load(self) -> None:
        """Load the .tflite model and label file. Called once."""

        log.info(f"Loading MobileNet SSD from: {self.cfg.MODEL_PATH}")
        self._interpreter = tf.lite.Interpreter(model_path=self.cfg.MODEL_PATH)
        self._interpreter.allocate_tensors()

        self._input_details  = self._interpreter.get_input_details()
        self._output_details = self._interpreter.get_output_details()

        # Read model's expected input size (may differ between variants)
        shape = self._input_details[0]["shape"]   # [1, H, W, 3]
        self._input_h, self._input_w = int(shape[1]), int(shape[2])

        self._labels = self._load_labels(self.cfg.LABEL_PATH)
        log.info(
            f"MobileNet SSD ready — input {self._input_w}×{self._input_h}, "
            f"{len(self._labels)} labels loaded."
        )
        self._loaded = True

    def run_inference(self, frame: np.ndarray) -> float:
        """
        Run MobileNet SSD person detection on one frame.

        Returns highest person confidence ≥ VISION_THRESHOLD, else 0.0.
        """
        self._ensure_loaded()

        # 1. Pre-process ────────────────────────────────────────────────
        input_tensor = self._preprocess(frame)

        # 2. Inference ──────────────────────────────────────────────────
        self._interpreter.set_tensor(
            self._input_details[0]["index"], input_tensor
        )
        self._interpreter.invoke()

        # 3. Read outputs ───────────────────────────────────────────────
        # Tensor index positions are model-specific; use [index] key.
        boxes   = self._interpreter.get_tensor(self._output_details[0]["index"])[0]
        classes = self._interpreter.get_tensor(self._output_details[1]["index"])[0]
        scores  = self._interpreter.get_tensor(self._output_details[2]["index"])[0]

        # 4. Filter person detections ───────────────────────────────────
        threshold = self.cfg.VISION_THRESHOLD
        best = 0.0

        for cls_id, score in zip(classes, scores):
            if score < threshold:
                continue
            if int(cls_id) == _PERSON_CLASS_ID:
                best = max(best, float(score))

        return best

    # ------------------------------------------------------------------ #
    #  Private helpers                                                    #
    # ------------------------------------------------------------------ #

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """
        BGR frame → (1, H, W, 3) uint8 RGB tensor ready for TFLite input.
        """
        resized = cv2.resize(frame, (self._input_w, self._input_h))
        rgb     = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        return np.expand_dims(rgb, axis=0).astype(np.uint8)

    def get_detections(self, frame: np.ndarray) -> List[dict]:
        """
        Extended method — returns ALL person detections above threshold
        as a list of dicts with bounding box coordinates.

        Used by visualisation/debug code; NOT part of BaseDetector contract.

        Returns:
            list of {
                "confidence": float,
                "box":  (x1, y1, x2, y2) in pixel coords,
                "label": "person",
            }
        """
        self._ensure_loaded()

        h, w = frame.shape[:2]
        input_tensor = self._preprocess(frame)

        self._interpreter.set_tensor(self._input_details[0]["index"], input_tensor)
        self._interpreter.invoke()

        boxes   = self._interpreter.get_tensor(self._output_details[0]["index"])[0]
        classes = self._interpreter.get_tensor(self._output_details[1]["index"])[0]
        scores  = self._interpreter.get_tensor(self._output_details[2]["index"])[0]

        threshold = self.cfg.VISION_THRESHOLD
        detections = []

        for i, (cls_id, score) in enumerate(zip(classes, scores)):
            if score < threshold:
                continue
            if int(cls_id) != _PERSON_CLASS_ID:
                continue

            ymin, xmin, ymax, xmax = boxes[i]
            detections.append({
                "confidence": float(score),
                "label":      "person",
                "box":        (
                    int(xmin * w),
                    int(ymin * h),
                    int(xmax * w),
                    int(ymax * h),
                ),
            })

        return detections