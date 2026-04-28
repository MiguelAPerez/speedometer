"""
pipeline.py — Single source of truth for pipeline component construction.

Both app.py and speed_detector.py import from here so default parameters
can never silently diverge between the two entry points.
"""

from __future__ import annotations

from .detector import Detector, DEFAULT_MODEL
from .tracker import CentroidTracker


def build_detector(
    model_path: str = DEFAULT_MODEL,
    conf_threshold: float = 0.40,
    iou_threshold: float = 0.35,
    min_area: int = 0,
) -> Detector:
    return Detector(
        model_path=model_path,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        min_area=min_area,
    )


def build_tracker(
    max_distance: float = 120.0,
    max_missing: int = 25,
    graveyard_max_frames: int = 150,
) -> CentroidTracker:
    return CentroidTracker(
        max_distance=max_distance,
        max_missing=max_missing,
        graveyard_max_frames=graveyard_max_frames,
    )
