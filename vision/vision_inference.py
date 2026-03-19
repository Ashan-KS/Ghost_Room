"""
vision/vision_inference.py — SACHITH
======================================
Loads the quantized MobileNet SSD model and runs person detection.

Your tasks:
  1. Load the .tflite model from config.MOBILENET_MODEL_PATH
  2. Implement run_inference() — returns a float confidence score 0.0–1.0
  3. Return 0.0 if no person detected, highest person confidence if detected

Download the model weights:
  wget https://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip
  unzip it and place the .tflite file at models/ssd_mobilenet_v2_coco_quant.tflite
"""

import logging
import numpy as np
import config
from vision.frame_utils import preprocess_frame

log = logging.getLogger(__name__)

_interpreter = None   # loaded once, reused


def _load_model():
    """Load TFLite model. Called once on first inference."""
    global _interpreter
    try:
        import tflite_runtime.interpreter as tflite
        _interpreter = tflite.Interpreter(model_path=config.MOBILENET_MODEL_PATH)
        _interpreter.allocate_tensors()
        log.info(f"MobileNet SSD loaded from {config.MOBILENET_MODEL_PATH}")
    except ImportError:
        log.warning("tflite_runtime not found — using stub inference (returns 0.0).")
    except Exception as e:
        log.error(f"Failed to load model: {e}")


def run_inference(frame) -> float:
    """
    Run person detection on a single frame.

    Args:
        frame: numpy array (H, W, 3) BGR image from OpenCV

    Returns:
        float: highest person confidence score (0.0–1.0).
                0.0 means no person detected above threshold.
    """
    # TODO (Sachith): implement this
    # Steps:
    #   1. preprocess_frame(frame) → (1, 300, 300, 3) tensor
    #   2. set_tensor on interpreter input
    #   3. invoke()
    #   4. get_tensor on output — boxes, classes, scores, count
    #   5. filter scores where class == 0 (person in COCO)
    #   6. return max person score, or 0.0 if none

    raise NotImplementedError("Sachith: implement run_inference() in vision_inference.py")
