"""
tracker.py — Kalman-filter centroid tracker.

Each track carries a constant-velocity Kalman filter that predicts the
centroid position forward each frame.  Incoming detections are matched
against *predicted* positions rather than last-known positions, so a car
that briefly disappears behind a tree/fence/pole can be re-matched as it
re-emerges, keeping the same track ID throughout.

Tracks that go unmatched for more than `max_missing` consecutive frames
are permanently dropped.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .detector import Detection, VEHICLE_LABELS


# ---------------------------------------------------------------------------
# Kalman filter factory
# ---------------------------------------------------------------------------

def _make_kalman(cx: float, cy: float) -> cv2.KalmanFilter:
    """
    4-D constant-velocity Kalman filter.
    State  : [cx, cy, vx, vy]
    Measure: [cx, cy]
    """
    kf = cv2.KalmanFilter(4, 2)
    kf.transitionMatrix = np.array([
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ], dtype=np.float32)
    kf.measurementMatrix = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
    ], dtype=np.float32)
    # Tuning: process noise (how much we trust the motion model) and
    # measurement noise (how much we trust the detector).
    kf.processNoiseCov     = np.eye(4, dtype=np.float32) * 0.05
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 4.0
    kf.errorCovPost        = np.eye(4, dtype=np.float32) * 10.0
    # Seed the state with the first detection
    kf.statePost = np.array([[cx], [cy], [0], [0]], dtype=np.float32)
    return kf


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class Track:
    track_id: int
    centroid: Tuple[float, float]          # last *detected* centroid
    predicted: Tuple[float, float]         # Kalman-predicted centroid
    bbox: Tuple[float, float, float, float]
    cls: int
    conf: float
    missing_frames: int = 0
    detected_this_frame: bool = True       # False when Kalman-predicted only
    history: List[Tuple[float, float]] = field(default_factory=list)
    _kf: Optional[object] = field(default=None, repr=False)

    @property
    def label(self) -> str:
        return VEHICLE_LABELS.get(self.cls, "vehicle")

    def predict(self) -> Tuple[float, float]:
        """Advance the Kalman filter one step and return predicted (cx, cy)."""
        pred = self._kf.predict()   # shape (4, 1)
        px, py = float(pred[0][0]), float(pred[1][0])
        self.predicted = (px, py)
        return px, py

    def correct(self, cx: float, cy: float) -> None:
        """Update the Kalman filter with an actual detection."""
        meas = np.array([[cx], [cy]], dtype=np.float32)
        self._kf.correct(meas)


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------

class CentroidTracker:
    """
    Kalman-filter nearest-neighbour tracker.

    Parameters
    ----------
    max_distance : float
        Max pixel distance between a detection centroid and a track's
        *predicted* position to count as a match.
    max_missing : int
        Frames a track can stay alive without a matching detection.
        Increase this to survive longer occlusions (behind trees etc.).
    """

    def __init__(self, max_distance: float = 120.0, max_missing: int = 25):
        self._next_id: int = 0
        self._tracks: Dict[int, Track] = OrderedDict()
        self.max_distance = max_distance
        self.max_missing = max_missing

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def tracks(self) -> Dict[int, Track]:
        return self._tracks

    def update(self, detections: List[Detection]) -> Dict[int, Track]:
        """
        Consume detections for the current frame.  Returns the updated
        dict of active tracks keyed by track ID.
        """
        # Step 1 — advance every Kalman filter one step
        predictions: Dict[int, Tuple[float, float]] = {}
        for tid, track in self._tracks.items():
            predictions[tid] = track.predict()

        # Mark all tracks as undetected this frame
        for track in self._tracks.values():
            track.detected_this_frame = False

        if not detections:
            self._age_and_prune()
            return self._tracks

        det_centroids = np.array([d.centroid for d in detections], dtype=float)

        if not self._tracks:
            for det in detections:
                self._register(det)
            return self._tracks

        # Step 2 — build distance matrix using *predicted* positions
        track_ids = list(self._tracks.keys())
        pred_centroids = np.array([predictions[tid] for tid in track_ids], dtype=float)
        dist_matrix = _pairwise_distances(pred_centroids, det_centroids)

        matched_tracks: set[int] = set()
        matched_dets:   set[int] = set()

        # Greedy match — smallest distance first
        flat = dist_matrix.copy()
        while True:
            if flat.size == 0:
                break
            min_val = flat.min()
            if min_val > self.max_distance:
                break
            t_idx, d_idx = np.unravel_index(flat.argmin(), flat.shape)
            tid = track_ids[t_idx]
            if tid not in matched_tracks and d_idx not in matched_dets:
                det = detections[d_idx]
                self._update_track(tid, det)
                matched_tracks.add(tid)
                matched_dets.add(d_idx)
            flat[t_idx, :] = np.inf
            flat[:, d_idx] = np.inf

        # Step 3 — register unmatched detections as new tracks
        for d_idx, det in enumerate(detections):
            if d_idx not in matched_dets:
                self._register(det)

        # Step 4 — age and prune unmatched tracks
        self._age_and_prune(matched=matched_tracks)

        return self._tracks

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _register(self, det: Detection) -> None:
        cx, cy = det.centroid
        kf = _make_kalman(cx, cy)
        t = Track(
            track_id=self._next_id,
            centroid=(cx, cy),
            predicted=(cx, cy),
            bbox=(det.x1, det.y1, det.x2, det.y2),
            cls=det.cls,
            conf=det.conf,
            detected_this_frame=True,
            _kf=kf,
        )
        t.history.append((cx, cy))
        self._tracks[self._next_id] = t
        self._next_id += 1

    def _update_track(self, tid: int, det: Detection) -> None:
        t = self._tracks[tid]
        cx, cy = det.centroid
        t.correct(cx, cy)
        t.centroid = (cx, cy)
        t.bbox = (det.x1, det.y1, det.x2, det.y2)
        t.cls = det.cls
        t.conf = det.conf
        t.missing_frames = 0
        t.detected_this_frame = True
        t.history.append((cx, cy))

    def _age_and_prune(self, matched: Optional[set] = None) -> None:
        matched = matched or set()
        stale = []
        for tid, t in self._tracks.items():
            if tid not in matched:
                t.missing_frames += 1
                if t.missing_frames > self.max_missing:
                    stale.append(tid)
        for tid in stale:
            del self._tracks[tid]


def _pairwise_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    diff = a[:, np.newaxis, :] - b[np.newaxis, :, :]
    return np.sqrt((diff ** 2).sum(axis=2))
