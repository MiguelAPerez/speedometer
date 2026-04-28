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
            if spd > 0.5:
                direction = ">>" if record.direction == "→" else "<<"
                label = f"#{tid} {spd:.1f} {units} {direction}"
            else:
                label = f"#{tid} stopped"
        else:
            label = f"#{tid} {track.label}"

        # Label background
        (tw, th), baseline = cv2.getTextSize(label, FONT, 0.55, 1)
        lx, ly = x1, max(y1 - 6, th + 4)
        cv2.rectangle(frame, (lx, ly - th - baseline - 2), (lx + tw + 4, ly + 2), colour, -1)
        cv2.putText(frame, label, (lx + 2, ly - baseline), FONT, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

    return frame


def draw_track(
    frame: np.ndarray,
    points: list,        # [{"x": float, "y": float}, ...]
    distances: list,     # [float, ...]  len = len(points) - 1
) -> np.ndarray:
    """
    Draw the calibration track onto a live frame.

    Each node is a labelled dot; segments are connected by lines with the
    real-world distance shown at the midpoint.
    """
    if not points:
        return frame
    h, w = frame.shape[:2]
    node_colours = [LINE_A_COLOUR, LINE_B_COLOUR, (200, 80, 255), (255, 200, 0), (0, 200, 255)]

    pts_px = [(int(p["x"]), int(p["y"])) for p in points]

    # Segment lines + distance labels
    for i, dist in enumerate(distances):
        x1, y1 = pts_px[i]
        x2, y2 = pts_px[i + 1]
        cv2.line(frame, (x1, y1), (x2, y2), (180, 180, 180), 1, cv2.LINE_AA)
        mx, my = (x1 + x2) // 2, (y1 + y2) // 2
        label = f"{dist:.1f}m"
        (tw, th), _ = cv2.getTextSize(label, FONT, 0.45, 1)
        cv2.rectangle(frame, (mx - 2, my - th - 2), (mx + tw + 2, my + 2), (40, 40, 40), -1)
        cv2.putText(frame, label, (mx, my), FONT, 0.45, (220, 220, 220), 1, cv2.LINE_AA)

    # Node dots
    labels = [chr(65 + i) for i in range(len(pts_px))]   # A, B, C, D ...
    for i, (cx, cy) in enumerate(pts_px):
        colour = node_colours[i % len(node_colours)]
        cv2.circle(frame, (cx, cy), 10, colour, 2, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), 3, colour, -1, cv2.LINE_AA)
        lbl = labels[i]
        flip = cx + 40 >= w
        if flip:
            cv2.line(frame, (cx - 22, cy), (cx - 10, cy), colour, 2, cv2.LINE_AA)
            cv2.putText(frame, lbl, (cx - 40, cy + 5), FONT, 0.55, colour, 1, cv2.LINE_AA)
        else:
            cv2.line(frame, (cx + 10, cy), (cx + 22, cy), colour, 2, cv2.LINE_AA)
            cv2.putText(frame, lbl, (cx + 26, cy + 5), FONT, 0.55, colour, 1, cv2.LINE_AA)

    return frame


# Alias for backwards compatibility
draw_calibration_lines = draw_track
draw_calibration_markers = draw_track


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
