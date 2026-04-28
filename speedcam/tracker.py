"""
tracker.py — Centroid nearest-neighbour tracker.

Each track is assigned a unique integer ID.  On every frame, incoming
detections are matched to existing tracks by minimum centroid distance.
Tracks that go unmatched for more than `max_missing` consecutive frames
are dropped.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .detector import Detection


@dataclass
class Track:
    track_id: int
    centroid: Tuple[float, float]         # (cx, cy) in pixels
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    cls: int
    conf: float
    missing_frames: int = 0
    history: List[Tuple[float, float]] = field(default_factory=list)  # centroid history


class CentroidTracker:
    """
    Nearest-neighbour centroid tracker.

    Parameters
    ----------
    max_distance : float
        Maximum pixel distance allowed when matching a detection to an
        existing track.  Detections beyond this threshold start a new track.
    max_missing : int
        Number of consecutive frames a track can go unmatched before it is
        permanently removed.
    """

    def __init__(self, max_distance: float = 80.0, max_missing: int = 5):
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
        if not detections:
            self._age_tracks()
            return self._tracks

        det_centroids = np.array([d.centroid for d in detections], dtype=float)

        if not self._tracks:
            # No existing tracks — register all detections
            for det in detections:
                self._register(det)
            return self._tracks

        track_ids = list(self._tracks.keys())
        track_centroids = np.array(
            [self._tracks[tid].centroid for tid in track_ids], dtype=float
        )

        # Pairwise distances: shape (n_tracks, n_detections)
        dist_matrix = _pairwise_distances(track_centroids, det_centroids)

        matched_tracks: set[int] = set()
        matched_dets: set[int] = set()

        # Greedy match: repeatedly take the smallest distance
        while True:
            if dist_matrix.size == 0:
                break
            min_val = dist_matrix.min()
            if min_val > self.max_distance:
                break
            t_idx, d_idx = np.unravel_index(dist_matrix.argmin(), dist_matrix.shape)
            tid = track_ids[t_idx]

            if tid not in matched_tracks and d_idx not in matched_dets:
                det = detections[d_idx]
                self._update_track(tid, det)
                matched_tracks.add(tid)
                matched_dets.add(d_idx)

            # Blank out this row/col so it isn't matched again
            dist_matrix[t_idx, :] = np.inf
            dist_matrix[:, d_idx] = np.inf

        # Age unmatched tracks
        for i, tid in enumerate(track_ids):
            if tid not in matched_tracks:
                self._tracks[tid].missing_frames += 1

        # Register new detections that had no match
        for d_idx, det in enumerate(detections):
            if d_idx not in matched_dets:
                self._register(det)

        # Remove stale tracks
        stale = [tid for tid, t in self._tracks.items() if t.missing_frames > self.max_missing]
        for tid in stale:
            del self._tracks[tid]

        return self._tracks

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _register(self, det: Detection) -> None:
        t = Track(
            track_id=self._next_id,
            centroid=det.centroid,
            bbox=(det.x1, det.y1, det.x2, det.y2),
            cls=det.cls,
            conf=det.conf,
        )
        t.history.append(det.centroid)
        self._tracks[self._next_id] = t
        self._next_id += 1

    def _update_track(self, tid: int, det: Detection) -> None:
        t = self._tracks[tid]
        t.centroid = det.centroid
        t.bbox = (det.x1, det.y1, det.x2, det.y2)
        t.cls = det.cls
        t.conf = det.conf
        t.missing_frames = 0
        t.history.append(det.centroid)

    def _age_tracks(self) -> None:
        stale = []
        for tid, t in self._tracks.items():
            t.missing_frames += 1
            if t.missing_frames > self.max_missing:
                stale.append(tid)
        for tid in stale:
            del self._tracks[tid]


def _pairwise_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Euclidean distance matrix between two sets of 2-D points."""
    diff = a[:, np.newaxis, :] - b[np.newaxis, :, :]   # (n_a, n_b, 2)
    return np.sqrt((diff ** 2).sum(axis=2))
