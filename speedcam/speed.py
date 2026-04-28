"""
speed.py — Per-frame displacement speed estimator.

The two calibration lines define a real-world scale:
    meters_per_pixel = distance_m / abs(line_b_y - line_a_y)

For every tracked vehicle we measure the pixel displacement of its
centroid between consecutive frames and convert to speed using the
elapsed wall-clock time between those frames.  A rolling window of
recent per-frame speed estimates is averaged for a smooth reading.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

from .tracker import Track


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class SpeedRecord:
    track_id: int
    speed_mps: float          # metres per second (smoothed)
    speed_mph: float
    speed_kph: float
    direction: str            # "→" or "←" (rightward / leftward motion)
    sample_count: int         # number of frames averaged


@dataclass
class _TrackState:
    """Internal per-track rolling state."""
    last_centroid: Optional[Tuple[float, float]] = None
    last_ts: Optional[float] = None
    speed_window: deque = field(default_factory=lambda: deque(maxlen=15))


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class SpeedEstimator:
    """
    Parameters
    ----------
    meters_per_pixel : float
        Scale derived from calibration: distance_m / pixel_span_of_lines.
    window : int
        Rolling window size for per-frame speed estimates (frames).
    min_samples : int
        Minimum frames before a speed reading is considered valid.
    """

    MPS_TO_MPH = 2.23694
    MPS_TO_KPH = 3.6

    def __init__(
        self,
        scale_pts: list,          # [(mid_y_px, mpp), ...] sorted by mid_y ascending
        window: int = 15,
        min_samples: int = 3,
        min_speed_mph: float = 2.0,
    ):
        # scale_pts is a list of (y_pixel, metres_per_pixel) control points.
        # A single entry means constant scale (backwards-compat / simple case).
        self._scale_pts = sorted(scale_pts, key=lambda p: p[0])
        self._window = window
        self._min_samples = min_samples
        self._min_speed_mps = min_speed_mph / self.MPS_TO_MPH
        self._states: Dict[int, _TrackState] = {}
        self._logged: Dict[int, SpeedRecord] = {}

    def _mpp_at(self, y: float) -> float:
        """Return interpolated metres-per-pixel at the given frame y-coordinate."""
        pts = self._scale_pts
        if len(pts) == 1:
            return pts[0][1]
        if y <= pts[0][0]:
            return pts[0][1]
        if y >= pts[-1][0]:
            return pts[-1][1]
        for i in range(len(pts) - 1):
            y0, m0 = pts[i]
            y1, m1 = pts[i + 1]
            if y0 <= y <= y1:
                t = (y - y0) / (y1 - y0)
                return m0 + t * (m1 - m0)
        return pts[-1][1]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self, tracks: Dict[int, Track], frame_ts: Optional[float] = None
    ) -> Dict[int, SpeedRecord]:
        """
        Process all active tracks for the current frame.

        Parameters
        ----------
        tracks : dict
            Active tracks from CentroidTracker.update().
        frame_ts : float, optional
            Timestamp for this frame (seconds).  Defaults to time.monotonic().

        Returns
        -------
        dict
            SpeedRecord for every track that has enough samples.
        """
        now = frame_ts if frame_ts is not None else time.monotonic()

        # Remove state for tracks that are no longer active
        active_ids = set(tracks.keys())
        stale = [tid for tid in self._states if tid not in active_ids]
        for tid in stale:
            del self._states[tid]

        results: Dict[int, SpeedRecord] = {}

        for tid, track in tracks.items():
            # Only use frames where the detector actually saw the vehicle —
            # skip Kalman-only predictions to avoid ghost displacement readings
            if not getattr(track, "detected_this_frame", True):
                continue
            if track.missing_frames > 0:
                continue

            state = self._states.setdefault(tid, _TrackState())
            cx, cy = track.centroid

            if state.last_centroid is not None and state.last_ts is not None:
                dt = now - state.last_ts
                if dt > 0:
                    dx = cx - state.last_centroid[0]
                    dy = cy - state.last_centroid[1]
                    displacement_px = (dx ** 2 + dy ** 2) ** 0.5
                    # Use scale at the car's current y-position
                    speed_mps = (displacement_px * self._mpp_at(cy)) / dt

                    # Spike rejection — discard readings that are more than 3×
                    # the rolling average, but only when the baseline is already
                    # above the minimum speed threshold.  Skipping rejection near
                    # zero prevents the window from blocking a car that starts
                    # stationary and then accelerates (every fast reading would
                    # be "> 3× ~0" and get dropped indefinitely).
                    if state.speed_window:
                        current_avg = sum(s for s, _ in state.speed_window) / len(state.speed_window)
                        if current_avg > self._min_speed_mps and speed_mps > current_avg * 3.0:
                            state.last_centroid = (cx, cy)
                            state.last_ts = now
                            continue

                    state.speed_window.append((speed_mps, dx))

            state.last_centroid = (cx, cy)
            state.last_ts = now

            if len(state.speed_window) >= self._min_samples:
                speeds = [s for s, _ in state.speed_window]
                dxs = [d for _, d in state.speed_window]
                avg_mps = sum(speeds) / len(speeds)
                avg_dx = sum(dxs) / len(dxs)
                direction = "→" if avg_dx >= 0 else "←"

                # Clamp to zero below the minimum threshold — suppresses
                # centroid jitter on parked / slow-moving vehicles
                if avg_mps < self._min_speed_mps:
                    avg_mps = 0.0

                record = SpeedRecord(
                    track_id=tid,
                    speed_mps=avg_mps,
                    speed_mph=avg_mps * self.MPS_TO_MPH,
                    speed_kph=avg_mps * self.MPS_TO_KPH,
                    direction=direction,
                    sample_count=len(state.speed_window),
                )
                self._logged[tid] = record
                results[tid] = record

        return results

    def get_record(self, track_id: int) -> Optional[SpeedRecord]:
        return self._logged.get(track_id)

    def reset(self) -> None:
        self._states.clear()
        self._logged.clear()

    @classmethod
    def from_track(cls, points: list, distances: list, **kwargs) -> "SpeedEstimator":
        """
        Build a depth-aware estimator from a multi-segment track calibration.

        points    — list of {x, y} dicts (pixel coordinates)
        distances — list of floats, len = len(points) - 1

        Each segment i connects points[i] → points[i+1] with a known
        real-world distance distances[i].  The scale (metres-per-pixel) is
        computed for each segment using its Euclidean pixel length, then
        stored at the segment midpoint y.  At runtime, speed.py interpolates
        between segments based on the vehicle's y-position in the frame.
        """
        if len(points) < 2 or len(distances) != len(points) - 1:
            raise ValueError("Need at least 2 points and one distance per segment")

        scale_pts = []
        for i, dist_m in enumerate(distances):
            p1, p2 = points[i], points[i + 1]
            span_px = ((p2["x"] - p1["x"]) ** 2 + (p2["y"] - p1["y"]) ** 2) ** 0.5
            if span_px < 1:
                raise ValueError(f"Segment {i} points are too close together")
            mpp = dist_m / span_px
            mid_y = (p1["y"] + p2["y"]) / 2
            scale_pts.append((mid_y, mpp))

        return cls(scale_pts=scale_pts, **kwargs)
