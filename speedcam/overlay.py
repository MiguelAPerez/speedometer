"""
overlay.py — Draw bounding boxes, track labels, calibration lines, and HUD.

All drawing is done in-place on the frame (BGR numpy array).
"""

from __future__ import annotations

from typing import Dict, Optional

import cv2
import numpy as np

from .tracker import Track
from .speed import SpeedRecord


# ---------------------------------------------------------------------------
# Colour palette (BGR)
# ---------------------------------------------------------------------------

COLOURS = [
    (0, 200, 255),   # yellow-orange
    (0, 255, 128),   # green
    (255, 100, 0),   # blue
    (200, 0, 255),   # purple
    (0, 128, 255),   # amber
    (255, 200, 0),   # cyan-ish
]

LINE_A_COLOUR = (0, 255, 180)    # teal
LINE_B_COLOUR = (0, 140, 255)    # orange
HUD_BG = (20, 20, 20)
HUD_TEXT = (240, 240, 240)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _track_colour(track_id: int) -> tuple:
    return COLOURS[track_id % len(COLOURS)]


# ---------------------------------------------------------------------------
# Public drawing functions
# ---------------------------------------------------------------------------

def draw_tracks(
    frame: np.ndarray,
    tracks: Dict[int, Track],
    speed_records: Dict[int, SpeedRecord],
    units: str = "mph",
) -> np.ndarray:
    """
    Draw bounding boxes and speed labels for all active tracks.

    Parameters
    ----------
    frame : np.ndarray
        BGR frame to annotate (modified in-place).
    tracks : dict
        Active tracks from CentroidTracker.
    speed_records : dict
        Speed estimates from SpeedEstimator.
    units : str
        "mph" or "kph"

    Returns
    -------
    np.ndarray
        The annotated frame (same object).
    """
    for tid, track in tracks.items():
        if track.missing_frames > 0:
            continue  # skip interpolated tracks

        colour = _track_colour(tid)
        x1, y1, x2, y2 = (int(v) for v in track.bbox)

        # Bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)

        # Build label string
        record = speed_records.get(tid)
        if record:
            spd = record.speed_mph if units == "mph" else record.speed_kph
            unit_str = units
            label = f"#{tid} {spd:.1f} {unit_str} {record.direction}"
        else:
            label = f"#{tid} {track.label}"

        # Label background
        (tw, th), baseline = cv2.getTextSize(label, FONT, 0.55, 1)
        lx, ly = x1, max(y1 - 6, th + 4)
        cv2.rectangle(frame, (lx, ly - th - baseline - 2), (lx + tw + 4, ly + 2), colour, -1)
        cv2.putText(frame, label, (lx + 2, ly - baseline), FONT, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

    return frame


def draw_calibration_lines(
    frame: np.ndarray,
    line_a_y: Optional[float],
    line_b_y: Optional[float],
) -> np.ndarray:
    """Draw the two horizontal calibration reference lines."""
    h, w = frame.shape[:2]
    if line_a_y is not None:
        y = int(line_a_y)
        cv2.line(frame, (0, y), (w, y), LINE_A_COLOUR, 2, cv2.LINE_AA)
        cv2.putText(frame, "A", (8, y - 6), FONT, 0.55, LINE_A_COLOUR, 1, cv2.LINE_AA)
    if line_b_y is not None:
        y = int(line_b_y)
        cv2.line(frame, (0, y), (w, y), LINE_B_COLOUR, 2, cv2.LINE_AA)
        cv2.putText(frame, "B", (8, y - 6), FONT, 0.55, LINE_B_COLOUR, 1, cv2.LINE_AA)
    return frame


def draw_hud(
    frame: np.ndarray,
    last_speed: Optional[float],
    avg_speed: Optional[float],
    count: int,
    units: str = "mph",
) -> np.ndarray:
    """
    Draw a semi-transparent HUD in the top-right corner showing:
      Last speed | Average speed | Vehicle count
    """
    h, w = frame.shape[:2]
    lines = [
        f"Last:  {last_speed:.1f} {units}" if last_speed is not None else "Last:  --",
        f"Avg:   {avg_speed:.1f} {units}" if avg_speed is not None else "Avg:   --",
        f"Count: {count}",
    ]

    padding = 8
    line_h = 22
    box_w = 210
    box_h = len(lines) * line_h + padding * 2
    x0, y0 = w - box_w - 10, 10

    # Semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + box_w, y0 + box_h), HUD_BG, -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    for i, line in enumerate(lines):
        cy = y0 + padding + (i + 1) * line_h - 4
        cv2.putText(frame, line, (x0 + padding, cy), FONT, 0.52, HUD_TEXT, 1, cv2.LINE_AA)

    return frame
