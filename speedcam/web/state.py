from __future__ import annotations

import threading
from typing import Optional

_lock = threading.Lock()

_state: dict = {
    "running": False,
    "source": None,
    "calibration": None,
    "units": "mph",
    "detections": [],
    "speeds_seen": [],
    "logged_ids": set(),
    "last_frame": None,       # latest annotated BGR frame (numpy array)
    "preview_frame": None,    # first frame for calibration preview
    "tmp_video_path": None,   # path to uploaded video temp file
    "error": None,
    "pipeline_thread": None,  # background threading.Thread
}


def reset_stats() -> None:
    with _lock:
        _state["detections"] = []
        _state["speeds_seen"] = []
        _state["logged_ids"] = set()
        _state["error"] = None
