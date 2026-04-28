"""
tracker.py — Kalman-filter centroid tracker with appearance-based re-identification.

Each track carries a constant-velocity Kalman filter that predicts the
centroid position forward each frame.  Incoming detections are matched
against *predicted* positions rather than last-known positions, so a car
that briefly disappears behind a tree/fence/pole can be re-matched as it
re-emerges, keeping the same track ID throughout.

When a track exceeds `max_missing` frames it moves to a *graveyard* rather
than being immediately deleted.  Graveyard tracks are held for up to
`graveyard_max_frames` additional frames.  Any new detection that cannot be
matched to an active track is compared to graveyard tracks using a combined
score of Kalman-predicted position and HSV color histogram similarity.  A
good match *resurrects* the old track ID — so the speed estimator's history
is preserved across the occlusion.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .detector import Detection, VEHICLE_LABELS
from .reid import extract_histogram, blend_histogram, histogram_distance


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
    kf.processNoiseCov     = np.eye(4, dtype=np.float32) * 0.05
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 4.0
    kf.errorCovPost        = np.eye(4, dtype=np.float32) * 10.0
    kf.statePost = np.array([[cx], [cy], [0], [0]], dtype=np.float32)
    return kf


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class Track:
    track_id: int
    centroid: Tuple[float, float]           # last *detected* centroid
    predicted: Tuple[float, float]          # Kalman-predicted centroid
    bbox: Tuple[float, float, float, float]
    cls: int
    conf: float
    missing_frames: int = 0
    detected_this_frame: bool = True        # False when Kalman-predicted only
    history: List[Tuple[float, float]] = field(default_factory=list)
    _kf: Optional[object] = field(default=None, repr=False)
    color_hist: Optional[np.ndarray] = field(default=None, repr=False)
    graveyard_frames: int = 0               # non-zero only while in graveyard

    @property
    def label(self) -> str:
        return VEHICLE_LABELS.get(self.cls, "vehicle")

    def predict(self) -> Tuple[float, float]:
        """Advance the Kalman filter one step and return predicted (cx, cy)."""
        pred = self._kf.predict()
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
    Kalman-filter nearest-neighbour tracker with graveyard re-identification.

    Parameters
    ----------
    max_distance : float
        Max pixel distance between a detection centroid and a track's
        predicted position to count as a match.
    max_missing : int
        Frames a track can stay alive without a matching detection before
        moving to the graveyard.
    graveyard_max_frames : int
        Frames a graveyard track is kept for re-identification attempts.
    reid_position_weight : float
        Weight of position score in the graveyard combined score (0–1).
    reid_appearance_weight : float
        Weight of appearance (histogram) score in the combined score (0–1).
    reid_score_threshold : float
        Combined score threshold for resurrection. Lower = stricter match.
    """

    def __init__(
        self,
        max_distance: float = 120.0,
        max_missing: int = 25,
        graveyard_max_frames: int = 150,
        reid_position_weight: float = 0.3,
        reid_appearance_weight: float = 0.7,
        reid_score_threshold: float = 0.45,
    ):
        self._next_id: int = 0
        self._tracks: Dict[int, Track] = OrderedDict()
        self._graveyard: Dict[int, Track] = OrderedDict()
        self.max_distance = max_distance
        self.max_missing = max_missing
        self.graveyard_max_frames = graveyard_max_frames
        self.reid_position_weight = reid_position_weight
        self.reid_appearance_weight = reid_appearance_weight
        self.reid_score_threshold = reid_score_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def tracks(self) -> Dict[int, Track]:
        return self._tracks

    def update(
        self,
        detections: List[Detection],
        frame: Optional[np.ndarray] = None,
    ) -> Dict[int, Track]:
        """
        Consume detections for the current frame.  Returns the updated
        dict of active tracks keyed by track ID.

        Parameters
        ----------
        detections : list of Detection
            Vehicle detections from the current frame.
        frame : np.ndarray, optional
            Raw BGR frame used to extract color histograms for ReID.
            If omitted, histogram-based re-identification is disabled but
            position-based graveyard matching still works.
        """
        # Step 1 — advance every Kalman filter one step (active + graveyard)
        predictions: Dict[int, Tuple[float, float]] = {}
        for tid, track in self._tracks.items():
            predictions[tid] = track.predict()
        for track in self._graveyard.values():
            track.predict()

        # Mark all active tracks as undetected this frame
        for track in self._tracks.values():
            track.detected_this_frame = False

        if not detections:
            self._age_and_prune()
            self._age_graveyard()
            return self._tracks

        det_centroids = np.array([d.centroid for d in detections], dtype=float)

        if not self._tracks:
            for det in detections:
                resurrected = self._match_graveyard(det, frame)
                if resurrected is not None:
                    self._tracks[resurrected.track_id] = resurrected
                else:
                    self._register(det, frame)
            self._age_graveyard()
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
                self._update_track(tid, det, frame)
                matched_tracks.add(tid)
                matched_dets.add(d_idx)
            flat[t_idx, :] = np.inf
            flat[:, d_idx] = np.inf

        # Step 3 — try graveyard resurrection before registering new tracks
        for d_idx, det in enumerate(detections):
            if d_idx not in matched_dets:
                resurrected = self._match_graveyard(det, frame)
                if resurrected is not None:
                    self._tracks[resurrected.track_id] = resurrected
                else:
                    self._register(det, frame)

        # Step 4 — age active tracks (unmatched ones move to graveyard)
        self._age_and_prune(matched=matched_tracks)

        # Step 5 — age graveyard
        self._age_graveyard()

        return self._tracks

    def reset(self) -> None:
        self._tracks.clear()
        self._graveyard.clear()
        self._next_id = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _register(self, det: Detection, frame: Optional[np.ndarray] = None) -> None:
        cx, cy = det.centroid
        kf = _make_kalman(cx, cy)
        color_hist = None
        if frame is not None:
            color_hist = extract_histogram(frame, (det.x1, det.y1, det.x2, det.y2))
        t = Track(
            track_id=self._next_id,
            centroid=(cx, cy),
            predicted=(cx, cy),
            bbox=(det.x1, det.y1, det.x2, det.y2),
            cls=det.cls,
            conf=det.conf,
            detected_this_frame=True,
            color_hist=color_hist,
            _kf=kf,
        )
        t.history.append((cx, cy))
        self._tracks[self._next_id] = t
        self._next_id += 1

    def _update_track(
        self,
        tid: int,
        det: Detection,
        frame: Optional[np.ndarray] = None,
    ) -> None:
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
        if frame is not None:
            new_hist = extract_histogram(frame, (det.x1, det.y1, det.x2, det.y2))
            if new_hist is not None:
                if t.color_hist is None:
                    t.color_hist = new_hist
                else:
                    t.color_hist = blend_histogram(t.color_hist, new_hist)

    def _age_and_prune(self, matched: Optional[set] = None) -> None:
        matched = matched or set()
        graduating = []
        for tid, t in self._tracks.items():
            if tid not in matched:
                t.missing_frames += 1
                if t.missing_frames > self.max_missing:
                    graduating.append(tid)
        for tid in graduating:
            track = self._tracks.pop(tid)
            track.graveyard_frames = 0
            self._graveyard[tid] = track

    def _age_graveyard(self) -> None:
        expired = []
        for tid, t in self._graveyard.items():
            t.graveyard_frames += 1
            if t.graveyard_frames > self.graveyard_max_frames:
                expired.append(tid)
        for tid in expired:
            del self._graveyard[tid]

    def _match_graveyard(
        self,
        det: Detection,
        frame: Optional[np.ndarray],
    ) -> Optional[Track]:
        """
        Try to match an unmatched detection against graveyard tracks using a
        combined position + appearance score.  Returns the resurrected Track
        (removed from graveyard, Kalman corrected) or None if no good match.
        """
        if not self._graveyard:
            return None

        det_hist = None
        if frame is not None:
            det_hist = extract_histogram(frame, (det.x1, det.y1, det.x2, det.y2))

        best_tid: Optional[int] = None
        best_score: float = float("inf")

        pos_norm = 3.0 * self.max_distance

        for tid, track in self._graveyard.items():
            dx = track.predicted[0] - det.centroid[0]
            dy = track.predicted[1] - det.centroid[1]
            pos_dist = (dx * dx + dy * dy) ** 0.5
            pos_score = min(pos_dist / pos_norm, 1.0)

            if det_hist is not None and track.color_hist is not None:
                app_score = histogram_distance(track.color_hist, det_hist)
            else:
                app_score = 0.5  # neutral when appearance data is unavailable

            combined = (
                self.reid_position_weight * pos_score
                + self.reid_appearance_weight * app_score
            )

            if combined < best_score:
                best_score = combined
                best_tid = tid

        if best_tid is None or best_score > self.reid_score_threshold:
            return None

        track = self._graveyard.pop(best_tid)
        track.missing_frames = 0
        track.graveyard_frames = 0
        track.detected_this_frame = True
        cx, cy = det.centroid
        track.centroid = (cx, cy)
        track.bbox = (det.x1, det.y1, det.x2, det.y2)
        track.cls = det.cls
        track.conf = det.conf
        track.history.append((cx, cy))
        track.correct(cx, cy)
        if det_hist is not None:
            if track.color_hist is None:
                track.color_hist = det_hist
            else:
                track.color_hist = blend_histogram(track.color_hist, det_hist)
        return track


def _pairwise_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    diff = a[:, np.newaxis, :] - b[np.newaxis, :, :]
    return np.sqrt((diff ** 2).sum(axis=2))
