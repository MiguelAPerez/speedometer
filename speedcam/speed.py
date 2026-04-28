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
        meters_per_pixel: float,
        window: int = 15,
        min_samples: int = 3,
    ):
        self._mpp = meters_per_pixel
        self._window = window
        self._min_samples = min_samples
        self._states: Dict[int, _TrackState] = {}
        self._logged: Dict[int, SpeedRecord] = {}  # last confirmed record per track

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
            if track.missing_frames > 0:
                # Don't compute speed on frames where the track was interpolated
                continue

            state = self._states.setdefault(tid, _TrackState())
            cx, cy = track.centroid

            if state.last_centroid is not None and state.last_ts is not None:
                dt = now - state.last_ts
                if dt > 0:
                    dx = cx - state.last_centroid[0]
                    dy = cy - state.last_centroid[1]
                    displacement_px = (dx ** 2 + dy ** 2) ** 0.5
                    speed_mps = (displacement_px * self._mpp) / dt
                    state.speed_window.append((speed_mps, dx))

            state.last_centroid = (cx, cy)
            state.last_ts = now

            if len(state.speed_window) >= self._min_samples:
                speeds = [s for s, _ in state.speed_window]
                dxs = [d for _, d in state.speed_window]
                avg_mps = sum(speeds) / len(speeds)
                avg_dx = sum(dxs) / len(dxs)
                direction = "→" if avg_dx >= 0 else "←"

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
    def from_calibration(cls, line_a_y: float, line_b_y: float, distance_m: float, **kwargs) -> "SpeedEstimator":
        """Convenience constructor: compute meters_per_pixel from calibration lines."""
        span_px = abs(line_b_y - line_a_y)
        if span_px == 0:
            raise ValueError("line_a_y and line_b_y must be different pixel positions")
        mpp = distance_m / span_px
        return cls(meters_per_pixel=mpp, **kwargs)
