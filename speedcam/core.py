"""
core.py — VideoSource abstraction and calibration I/O.

Calibration is stored in calibration.json keyed by a source identifier:
  - webcam:0  (or webcam:1 etc.)
  - absolute file path
  - rtsp/http URL

Each entry stores:
  {
    "line_a_y": float,
    "line_b_y": float,
    "distance_m": float,
    "frame_w": int,
    "frame_h": int,
    "created_at": ISO-8601 string
  }

If the actual frame size differs from the saved size, line positions are
scaled proportionally.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

CALIBRATION_FILE = Path("calibration.json")


# ---------------------------------------------------------------------------
# Calibration helpers
# ---------------------------------------------------------------------------

def _load_json() -> dict:
    if CALIBRATION_FILE.exists():
        try:
            return json.loads(CALIBRATION_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_json(data: dict) -> None:
    CALIBRATION_FILE.write_text(json.dumps(data, indent=2))


def source_key(source: str | int) -> str:
    """Return a stable string key for the given source."""
    if isinstance(source, int):
        return f"webcam:{source}"
    src = str(source)
    # Webcam paths like /dev/video0 or "0"
    if src.isdigit():
        return f"webcam:{src}"
    # Absolute path — normalise
    if os.path.exists(src):
        return str(Path(src).resolve())
    return src  # URL or other identifier


def load_calibration(source: str | int, frame_w: int, frame_h: int) -> Optional[dict]:
    """
    Load saved calibration for *source*.

    If saved frame size differs from *frame_w* / *frame_h*, the line
    positions are scaled proportionally before being returned.

    Returns None if no calibration is found for this source.
    """
    key = source_key(source)
    data = _load_json()
    entry = data.get(key)
    if entry is None:
        return None

    saved_w = entry.get("frame_w", frame_w)
    saved_h = entry.get("frame_h", frame_h)

    line_a_y = entry["line_a_y"]
    line_b_y = entry["line_b_y"]

    # Scale if frame size changed
    if saved_h != frame_h:
        scale = frame_h / saved_h
        line_a_y = line_a_y * scale
        line_b_y = line_b_y * scale

    return {
        "line_a_y": line_a_y,
        "line_b_y": line_b_y,
        "distance_m": entry["distance_m"],
        "frame_w": frame_w,
        "frame_h": frame_h,
    }


def save_calibration(
    source: str | int,
    line_a_y: float,
    line_b_y: float,
    distance_m: float,
    frame_w: int,
    frame_h: int,
) -> None:
    """Persist calibration for *source* to calibration.json."""
    key = source_key(source)
    data = _load_json()
    data[key] = {
        "line_a_y": line_a_y,
        "line_b_y": line_b_y,
        "distance_m": distance_m,
        "frame_w": frame_w,
        "frame_h": frame_h,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _save_json(data)


# ---------------------------------------------------------------------------
# VideoSource
# ---------------------------------------------------------------------------

class VideoSource:
    """
    Unified interface over webcam index, video file path, or RTSP/HTTP URL.

    Usage
    -----
        with VideoSource(0) as vs:
            while True:
                ok, frame = vs.read()
                if not ok:
                    break

    Attributes
    ----------
    width, height : int
        Frame dimensions of the source.
    fps : float
        Reported FPS (may be 0 for RTSP; use wall-clock timing in that case).
    source_key : str
        Stable string identifier used for calibration lookups.
    """

    # Max reconnect attempts for streaming sources
    _MAX_RECONNECTS = 5
    _RECONNECT_DELAY = 2.0  # seconds

    def __init__(self, source: str | int, source_fps_override: Optional[float] = None):
        self._source = source
        self._fps_override = source_fps_override
        self._cap: Optional[cv2.VideoCapture] = None
        self._reconnects = 0
        self.source_key = source_key(source)
        self._open()

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> "VideoSource":
        return self

    def __exit__(self, *_) -> None:
        self.release()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read the next frame.  Returns (True, frame) on success, or
        (False, None) when the source is exhausted or unrecoverable.
        """
        if self._cap is None:
            return False, None

        ok, frame = self._cap.read()
        if not ok:
            if self._is_streaming():
                return self._try_reconnect()
            return False, None
        return True, frame

    def grab_single_frame(self) -> Optional[np.ndarray]:
        """Return a single frame for preview/calibration purposes."""
        ok, frame = self.read()
        return frame if ok else None

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    @property
    def width(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if self._cap else 0

    @property
    def height(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if self._cap else 0

    @property
    def fps(self) -> float:
        if self._fps_override is not None:
            return self._fps_override
        reported = self._cap.get(cv2.CAP_PROP_FPS) if self._cap else 0.0
        return reported if reported > 0 else 30.0  # sane default

    @property
    def frame_count(self) -> int:
        """Total frame count for file sources; -1 for live sources."""
        if self._cap is None:
            return -1
        count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return count if count > 0 else -1

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _open(self) -> None:
        src = self._source if not isinstance(self._source, str) or not self._source.isdigit() \
              else int(self._source)
        self._cap = cv2.VideoCapture(src)
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open video source: {self._source!r}")

    def _is_streaming(self) -> bool:
        src = str(self._source)
        return src.startswith("rtsp://") or src.startswith("http://") or src.startswith("https://")

    def _try_reconnect(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self._reconnects >= self._MAX_RECONNECTS:
            return False, None
        self._reconnects += 1
        time.sleep(self._RECONNECT_DELAY)
        try:
            self.release()
            self._open()
            ok, frame = self._cap.read()
            if ok:
                self._reconnects = 0
            return ok, frame if ok else None
        except RuntimeError:
            return False, None
