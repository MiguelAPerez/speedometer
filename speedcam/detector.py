"""
detector.py — YOLOv8-nano wrapper for vehicle detection.

Returns per-frame detections filtered to COCO vehicle classes:
  2 = car, 3 = motorcycle, 5 = bus, 7 = truck
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

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
    Thin wrapper around YOLOv8-nano.  The model is loaded once and reused
    across frames.  Only vehicle-class detections above `conf_threshold`
    are returned.
    """

    def __init__(self, model_path: str = "yolov8n.pt", conf_threshold: float = 0.35):
        from ultralytics import YOLO  # deferred so import errors surface clearly

        self._model = YOLO(model_path)
        self._conf = conf_threshold

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Run inference on a single BGR frame (as returned by cv2.VideoCapture).
        Returns a list of Detection objects for every vehicle found.
        """
        results = self._model(frame, verbose=False)[0]
        detections: List[Detection] = []

        for box in results.boxes:
            cls = int(box.cls[0])
            if cls not in VEHICLE_CLASSES:
                continue
            conf = float(box.conf[0])
            if conf < self._conf:
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append(Detection(x1=x1, y1=y1, x2=x2, y2=y2, conf=conf, cls=cls))

        return detections
