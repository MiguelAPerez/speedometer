"""
detector.py — YOLO vehicle detector wrapper.

Returns per-frame detections filtered to COCO vehicle classes:
  2 = car, 3 = motorcycle, 5 = bus, 7 = truck
"""

from __future__ import annotations

import platform
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

DEFAULT_MODEL = "yolo12s.pt"


def _best_device() -> str:
    """Return the best available inference device: cuda > mps > cpu."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if platform.system() == "Darwin":
            mps = getattr(torch.backends, "mps", None)
            if mps is not None and mps.is_available():
                return "mps"
    except Exception:
        pass
    return "cpu"

# COCO class IDs we care about
VEHICLE_CLASSES = {2, 3, 5, 7}
VEHICLE_LABELS = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


@dataclass
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float
    cls: int

    @property
    def centroid(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def label(self) -> str:
        return VEHICLE_LABELS.get(self.cls, "vehicle")


class Detector:
    """
    Thin wrapper around a YOLO model. The model is loaded once and reused
    across frames. Only vehicle-class detections above `conf_threshold`
    and `min_area` are returned.

    Parameters
    ----------
    model_path : str
        Ultralytics model identifier or local .pt path. Auto-downloaded if absent.
    conf_threshold : float
        Minimum detection confidence (0–1). Raise to suppress false positives
        like vegetation or shadows.
    iou_threshold : float
        NMS IoU threshold passed to YOLO. Lower values merge overlapping boxes
        more aggressively, preventing one car from generating two detections.
    min_area : int
        Minimum bounding-box area in pixels. Filters detections that are too
        small to be a real vehicle at the expected scene scale. 0 = disabled.
    """

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL,
        conf_threshold: float = 0.40,
        iou_threshold: float = 0.35,
        min_area: int = 0,
        device: Optional[str] = None,
        imgsz: int = 640,
    ):
        from ultralytics import YOLO  # deferred so import errors surface clearly

        self._conf = conf_threshold
        self._iou = iou_threshold
        self._min_area = min_area
        self._imgsz = imgsz
        self._device = device if device is not None else _best_device()
        # FP16 only on CUDA — MPS half support varies across PyTorch versions
        self._half = self._device == "cuda"
        self._model = YOLO(model_path)
        self._model.to(self._device)
        print(f"[detector] device={self._device}  half={self._half}  imgsz={self._imgsz}")

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Run inference on a single BGR frame (as returned by cv2.VideoCapture).
        Returns a list of Detection objects for every vehicle found.
        """
        results = self._model(
            frame, verbose=False, iou=self._iou,
            half=self._half, imgsz=self._imgsz,
        )[0]
        detections: List[Detection] = []

        for box in results.boxes:
            cls = int(box.cls[0])
            if cls not in VEHICLE_CLASSES:
                continue
            conf = float(box.conf[0])
            if conf < self._conf:
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            if self._min_area > 0:
                if (x2 - x1) * (y2 - y1) < self._min_area:
                    continue
            detections.append(Detection(x1=x1, y1=y1, x2=x2, y2=y2, conf=conf, cls=cls))

        return detections
