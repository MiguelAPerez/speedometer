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


def get_data_dir() -> Path:
    """User-writable directory for calibration.json and temp uploads.

    Defaults to the current working directory. Set SPEEDCAM_DATA_DIR to
    override — the packaged launcher points this at ~/.speedcam so calibration
    is never written inside the read-only app bundle.
    """
    env = os.environ.get("SPEEDCAM_DATA_DIR")
    return Path(env) if env else Path(".")


# ---------------------------------------------------------------------------
# Calibration helpers
# ---------------------------------------------------------------------------

def _cal_path() -> Path:
    return get_data_dir() / "calibration.json"


def _load_json() -> dict:
    path = _cal_path()
    if path.exists():
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_json(data: dict) -> None:
    _cal_path().write_text(json.dumps(data, indent=2))


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


def is_live_camera(source: str | int) -> bool:
    """Return True only for webcam/device-index sources (not files or URLs)."""
    if isinstance(source, int):
        return True
    src = str(source)
    return src.isdigit()


def clear_calibration(source: str | int) -> None:
    """Remove any saved calibration entry for *source* from calibration.json."""
    key = source_key(source)
    data = _load_json()
    if key in data:
        del data[key]
        _save_json(data)


def load_calibration(source: str | int, frame_w: int, frame_h: int) -> Optional[dict]:
    """
    Load saved track calibration for *source*.

    Returns a dict with keys:
      points    — list of {x, y} dicts (pixel coords scaled to current frame)
      distances — list of floats (real-world metres per segment)
      frame_w, frame_h

    Returns None if no valid calibration exists for this source.
    """
    key = source_key(source)
    data = _load_json()
    entry = data.get(key)
    if entry is None:
        return None

    # Must be the track format
    if "points" not in entry or "distances" not in entry:
        return None

    saved_w = entry.get("frame_w", frame_w)
    saved_h = entry.get("frame_h", frame_h)
    scale_x = frame_w / saved_w if saved_w else 1.0
    scale_y = frame_h / saved_h if saved_h else 1.0

    points = [
        {"x": p["x"] * scale_x, "y": p["y"] * scale_y}
        for p in entry["points"]
    ]

    return {
        "points": points,
        "distances": entry["distances"],
        "frame_w": frame_w,
        "frame_h": frame_h,
    }


def save_calibration(
    source: str | int,
    points: list,
    distances: list,
    frame_w: int,
    frame_h: int,
) -> None:
    """
    Persist track calibration for *source* to calibration.json.

    points    — list of {x, y} dicts
    distances — list of floats, length = len(points) - 1
    """
    key = source_key(source)
    data = _load_json()
    data[key] = {
        "points": points,
        "distances": distances,
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
        # Request hardware-accelerated decode when the platform supports it
        # (NVDEC on NVIDIA, VideoToolbox on macOS, VAAPI on Linux).
        # Silently ignored on builds or platforms where the property is absent.
        _hw_prop = getattr(cv2, "CAP_PROP_HW_ACCELERATION", None)
        _hw_any  = getattr(cv2, "VIDEO_ACCELERATION_ANY", None)
        if _hw_prop is not None and _hw_any is not None:
            try:
                self._cap.set(_hw_prop, _hw_any)
            except Exception:
                pass
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
