"""
vision/yolo_detector.py
========================
Concrete detector backend: YOLOv8 (or any Ultralytics YOLO) via the
ultralytics Python package, with an optional ONNX Runtime fallback
for environments where the full package is too heavy (e.g. bare Pi).

Model file expected at:  config.MODEL_PATH
Label file expected at:  config.LABEL_PATH   (optional — YOLO embeds its own)

Two sub-backends, selected automatically:
    1. ultralytics  — best for development; auto-downloads weights on first run.
    2. onnxruntime  — lightweight; requires exporting the model first:
                      yolo export model=yolov8n.pt format=onnx

Drop your weights in models/:
    models/yolov8n.pt          ← Ultralytics native (recommended for dev)
    models/yolov8n.onnx        ← ONNX export (recommended for Pi deployment)

COCO class id for "person" = 0   (same as MobileNet SSD).
"""

import logging
import numpy as np
import cv2

from vision.base_detector import BaseDetector

log = logging.getLogger(__name__)

_PERSON_CLASS_ID = 0       # COCO index — identical across both model families


class YoloDetector(BaseDetector):
    """
    YOLOv8 person detector.

    Tries ultralytics first, falls back to onnxruntime if unavailable.
    Both sub-backends expose the identical BaseDetector interface so
    no caller code changes when you switch.
    """

    def _init_(self, cfg):
        super().__init__(cfg)
        self._model   = None        # ultralytics YOLO object  OR  ort.InferenceSession
        self._backend = None        # "ultralytics" | "onnx"
        self._input_h = 640         # default YOLO input size
        self._input_w = 640
        self._labels: list[str] = []

    # ------------------------------------------------------------------ #
    #  BaseDetector interface                                             #
    # ------------------------------------------------------------------ #

    def load(self) -> None:
        """Load YOLO weights. Checks if .onnx to use onnxruntime directly, else ultralytics."""
        model_path = self.cfg.MODEL_PATH
        # If .onnx is specified, prefer onnxruntime first to avoid PyTorch SIGILL on Pi
        if model_path.endswith(".onnx") and self._try_load_onnx(model_path):
            self._backend = "onnx"
        elif self._try_load_ultralytics(model_path):
            self._backend = "ultralytics"
        elif self._try_load_onnx(model_path):
            self._backend = "onnx"
        else:
            raise RuntimeError(
                "Could not load YOLO model. Install either:\n"
                "  pip install ultralytics      (recommended)\n"
                "  pip install onnxruntime      (lightweight Pi option)\n"
                f"Model path: {model_path}"
            )
        # Labels: YOLO embeds its own, but honour an external file if provided
        try:
            self._labels = self._load_labels(self.cfg.LABEL_PATH)
        except (FileNotFoundError, AttributeError):
            # Fine — we'll use the class-id integer directly
            self._labels = []
        self._loaded = True
        log.info(f"YOLO detector ready — backend={self._backend}, model={model_path}")
        
    def run_inference(self, frame: np.ndarray) -> float:
        """
        Run YOLO person detection on one frame.
        Returns highest person confidence ≥ VISION_THRESHOLD, else 0.0.
        """
        self._ensure_loaded()
        if self._backend == "ultralytics":
            return self._infer_ultralytics(frame)
        else:
            return self._infer_onnx(frame)

    # ------------------------------------------------------------------ #
    #  Ultralytics sub-backend                                           #
    # ------------------------------------------------------------------ #

    def _try_load_ultralytics(self, model_path: str) -> bool:
        try:
            from ultralytics import YOLO  # noqa: PLC0415
            self._model = YOLO(model_path)
            log.info(f"Loaded via ultralytics: {model_path}")
            return True
        except ImportError:
            log.debug("ultralytics not installed — skipping.")
            return False
        except Exception as exc:
            log.warning(f"ultralytics load failed: {exc}")
            return False

    def _infer_ultralytics(self, frame: np.ndarray) -> float:
        """Run inference using the ultralytics YOLO API."""
        threshold = self.cfg.VISION_THRESHOLD

        # ultralytics accepts BGR numpy arrays directly
        results = self._model(
            frame,
            verbose=False,
            show=False,        # ← stops ultralytics opening its own window
            visualize=False,   # ← stops it rendering feature map windows
            conf=threshold,
            classes=[_PERSON_CLASS_ID],   # only detect persons → faster
        )

        best = 0.0
        for result in results:
            if result.boxes is None:
                continue
            for conf_tensor in result.boxes.conf:
                best = max(best, float(conf_tensor))

        return best

    # ------------------------------------------------------------------ #
    #  ONNX Runtime sub-backend                                          #
    # ------------------------------------------------------------------ #

    def _try_load_onnx(self, model_path: str) -> bool:
        if not model_path.endswith(".onnx"):
            log.debug("Model path doesn't end with .onnx — skipping ONNX backend.")
            return False
        try:
            import onnxruntime as ort  # noqa: PLC0415
            providers = ["CPUExecutionProvider"]
            self._model = ort.InferenceSession(model_path, providers=providers)
            # Read expected input shape from the model graph
            inp = self._model.get_inputs()[0]
            _, _, h, w = inp.shape          # NCHW
            self._input_h, self._input_w = int(h), int(w)
            log.info(f"Loaded via onnxruntime: {model_path}  ({w}×{h})")
            return True
        except ImportError:
            log.debug("onnxruntime not installed — skipping.")
            return False
        except Exception as exc:
            log.warning(f"ONNX load failed: {exc}")
            return False

    def _infer_onnx(self, frame: np.ndarray) -> float:
        """
        Run inference using ONNX Runtime.

        YOLOv8 ONNX output shape: (1, 84, N_anchors)
        Rows 0-3 = cx, cy, w, h  (normalised)
        Rows 4+  = class scores (80 COCO classes)
        Person score = row 4.
        """
        threshold = self.cfg.VISION_THRESHOLD

        input_tensor = self._preprocess_onnx(frame)
        input_name   = self._model.get_inputs()[0].name
        outputs      = self._model.run(None, {input_name: input_tensor})

        # outputs[0] shape: (1, 84, num_anchors)
        preds = outputs[0][0]                   # (84, num_anchors)
        class_scores = preds[4:, :]             # (80, num_anchors)
        person_scores = class_scores[_PERSON_CLASS_ID, :]  # (num_anchors,)

        above = person_scores[person_scores >= threshold]
        return float(above.max()) if len(above) > 0 else 0.0

    def _preprocess_onnx(self, frame: np.ndarray) -> np.ndarray:
        """BGR frame → (1, 3, H, W) float32 NCHW tensor, values 0–1."""
        resized  = cv2.resize(frame, (self._input_w, self._input_h))
        rgb      = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        chw      = np.transpose(rgb, (2, 0, 1)).astype(np.float32) / 255.0
        return np.expand_dims(chw, axis=0)

    # ------------------------------------------------------------------ #
    #  Extended helper — returns full detection list (not part of        #
    #  BaseDetector, same interface as MobileNetDetector.get_detections) #
    # ------------------------------------------------------------------ #

    def get_detections(self, frame: np.ndarray) -> list[dict]:
        """
        Returns all person detections above threshold as a list of dicts:
            {"confidence": float, "box": (x1, y1, x2, y2), "label": "person"}
        """
        self._ensure_loaded()
        if self._backend == "ultralytics":
            return self._detections_ultralytics(frame)
        return self._detections_onnx(frame)

    def _detections_ultralytics(self, frame: np.ndarray) -> list[dict]:
        threshold = self.cfg.VISION_THRESHOLD
        results   = self._model(
            frame, verbose=False, conf=threshold,
            classes=[_PERSON_CLASS_ID]
        )
        detections = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
                detections.append({
                    "confidence": float(box.conf[0]),
                    "label":      "person",
                    "box":        (x1, y1, x2, y2),
                })
        return detections

    def _detections_onnx(self, frame: np.ndarray) -> list[dict]:
        h, w     = frame.shape[:2]
        threshold = self.cfg.VISION_THRESHOLD
        input_tensor = self._preprocess_onnx(frame)
        input_name   = self._model.get_inputs()[0].name
        outputs      = self._model.run(None, {input_name: input_tensor})
        preds        = outputs[0][0]           # (84, num_anchors)
        cx, cy, bw, bh = preds[0], preds[1], preds[2], preds[3]
        person_scores  = preds[4 + _PERSON_CLASS_ID]
        boxes_nms = []
        scores_nms = []
        for i, score in enumerate(person_scores):
            if score < threshold:
                continue
            x_left = int((cx[i] - bw[i] / 2) * w)
            y_top  = int((cy[i] - bh[i] / 2) * h)
            w_px   = int(bw[i] * w)
            h_px   = int(bh[i] * h)
            
            boxes_nms.append([x_left, y_top, w_px, h_px])
            scores_nms.append(float(score))
        detections = []
        if len(boxes_nms) > 0:
            # OpenCV NMS to eliminate multiple overlapping boxes for the same person
            indices = cv2.dnn.NMSBoxes(
                boxes_nms, 
                scores_nms, 
                score_threshold=threshold, 
                nms_threshold=0.45
            )
            
            if len(indices) > 0:
                for i in indices.flatten():
                    x, y, w_box, h_box = boxes_nms[i]
                    detections.append({
                        "confidence": scores_nms[i],
                        "label":      "person",
                        "box":        (x, y, x + w_box, y + h_box),
                    })
        return detections
