#!/usr/bin/env python3
"""
speed_detector.py — CLI entry point for the speedcam system.

Usage examples
--------------
  # Live webcam, mph
  python speed_detector.py --source 0 --units mph

  # Video file, prompt calibration, record output
  python speed_detector.py --source clip.mp4 --calibrate --record

  # RTSP stream, km/h
  python speed_detector.py --source rtsp://192.168.1.10:8554/cam --units kph

Keyboard shortcuts (live window)
---------------------------------
  q  — quit
  r  — reset stats (keep calibration)
  s  — save current frame as screenshot.png
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from speedcam.core import VideoSource, load_calibration, save_calibration
from speedcam.detector import Detector
from speedcam.tracker import CentroidTracker
from speedcam.speed import SpeedEstimator, SpeedRecord
from speedcam.overlay import draw_tracks, draw_calibration_lines, draw_hud


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

CSV_COLUMNS = [
    "timestamp", "track_id", "vehicle_type",
    "speed_mph", "speed_kph", "direction", "sample_count",
]


def _open_csv(path: str):
    """Open (or append to) detections.csv and return (file_handle, writer)."""
    exists = Path(path).exists()
    fh = open(path, "a", newline="")
    writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
    if not exists:
        writer.writeheader()
    return fh, writer


def _write_row(writer, track, record: SpeedRecord) -> None:
    writer.writerow({
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "track_id": record.track_id,
        "vehicle_type": track.label,
        "speed_mph": round(record.speed_mph, 2),
        "speed_kph": round(record.speed_kph, 2),
        "direction": record.direction,
        "sample_count": record.sample_count,
    })


# ---------------------------------------------------------------------------
# Calibration UI (OpenCV mouse callback)
# ---------------------------------------------------------------------------

def run_calibration_ui(frame: np.ndarray, source) -> Optional[dict]:
    """
    Show the first frame in an OpenCV window.  User clicks twice to set
    Line A and Line B (y-coordinates only), then enters the real-world
    distance in metres via the terminal.

    Returns a calibration dict or None if the user cancelled.
    """
    clone = frame.copy()
    h, w = clone.shape[:2]
    clicks: List[int] = []  # y-coordinates

    FONT = cv2.FONT_HERSHEY_SIMPLEX

    def _redraw():
        img = clone.copy()
        instructions = [
            "Click to set Line A (top reference)",
            "Click to set Line B (bottom reference)",
        ]
        msg_idx = min(len(clicks), 1)
        if len(clicks) < 2:
            cv2.putText(img, instructions[msg_idx], (10, 30), FONT, 0.7, (0, 255, 200), 2)
        for i, y in enumerate(clicks):
            colour = (0, 255, 180) if i == 0 else (0, 140, 255)
            label = "A" if i == 0 else "B"
            cv2.line(img, (0, y), (w, y), colour, 2)
            cv2.putText(img, label, (10, y - 8), FONT, 0.65, colour, 2)
        if len(clicks) == 2:
            cv2.putText(img, "Lines set. Enter distance in terminal.", (10, 30), FONT, 0.65, (255, 255, 100), 2)
        cv2.imshow("Calibration — speedcam", img)

    def _on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < 2:
            clicks.append(y)
            _redraw()

    cv2.namedWindow("Calibration — speedcam", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Calibration — speedcam", _on_mouse)
    _redraw()

    print("\n[Calibration] Click Line A on the preview window, then Line B.")
    print("Press ESC to cancel.\n")

    while True:
        key = cv2.waitKey(50) & 0xFF
        if key == 27:  # ESC
            cv2.destroyWindow("Calibration — speedcam")
            return None
        if len(clicks) == 2:
            break

    cv2.destroyWindow("Calibration — speedcam")

    # Get distance from terminal
    while True:
        raw = input("Enter real-world distance between the two lines (metres): ").strip()
        try:
            distance_m = float(raw)
            if distance_m <= 0:
                raise ValueError
            break
        except ValueError:
            print("  Please enter a positive number (e.g. 10.0)")

    calib = {
        "line_a_y": float(clicks[0]),
        "line_b_y": float(clicks[1]),
        "distance_m": distance_m,
        "frame_w": w,
        "frame_h": h,
    }
    save_calibration(source, **{k: v for k, v in calib.items()})
    print(f"[Calibration] Saved — Line A y={clicks[0]}px, Line B y={clicks[1]}px, {distance_m}m\n")
    return calib


# ---------------------------------------------------------------------------
# Main processing loop
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    source = args.source
    # Coerce numeric string to int so VideoCapture treats it as a device index
    if isinstance(source, str) and source.isdigit():
        source = int(source)

    print(f"[speedcam] Opening source: {source!r}")
    vs = VideoSource(source, source_fps_override=args.source_fps)
    w, h = vs.width, vs.height
    print(f"[speedcam] Frame size: {w}×{h}  FPS: {vs.fps:.1f}")

    # ---- Calibration ----
    calib = None
    if not args.calibrate:
        calib = load_calibration(source, w, h)
        if calib:
            print(f"[speedcam] Loaded saved calibration for this source.")

    if calib is None:
        first_frame = vs.grab_single_frame()
        if first_frame is None:
            print("[speedcam] ERROR: Could not read first frame for calibration.")
            vs.release()
            sys.exit(1)
        calib = run_calibration_ui(first_frame, source)
        if calib is None:
            print("[speedcam] Calibration cancelled.")
            vs.release()
            sys.exit(0)

    line_a_y = calib["line_a_y"]
    line_b_y = calib["line_b_y"]
    distance_m = calib["distance_m"]

    # ---- Pipeline ----
    detector = Detector(conf_threshold=args.conf)
    tracker = CentroidTracker(max_distance=args.max_distance, max_missing=args.max_missing)
    estimator = SpeedEstimator.from_calibration(line_a_y, line_b_y, distance_m)

    # ---- Output files ----
    csv_fh, csv_writer = _open_csv("detections.csv")
    logged_ids: set[int] = set()

    writer_out: Optional[cv2.VideoWriter] = None
    if args.record:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer_out = cv2.VideoWriter("output.mp4", fourcc, vs.fps, (w, h))
        print("[speedcam] Recording to output.mp4")

    # ---- Stats ----
    speeds_seen: List[float] = []

    # ---- Frame skip ----
    skip = max(1, args.skip_frames)
    frame_idx = 0

    print("[speedcam] Running — press q to quit, r to reset stats, s for screenshot\n")

    try:
        while True:
            frame_ts = time.monotonic()
            ok, frame = vs.read()
            if not ok:
                print("[speedcam] End of source.")
                break

            frame_idx += 1

            # Run detection only every `skip` frames; tracker still gets updated
            if frame_idx % skip == 0:
                detections = detector.detect(frame)
            else:
                detections = []

            tracks = tracker.update(detections)
            speed_records = estimator.update(tracks, frame_ts=frame_ts)

            # Log new records to CSV
            for tid, record in speed_records.items():
                if tid not in logged_ids:
                    track = tracks.get(tid)
                    if track:
                        _write_row(csv_writer, track, record)
                        csv_fh.flush()
                        logged_ids.add(tid)
                        speed_val = record.speed_mph if args.units == "mph" else record.speed_kph
                        speeds_seen.append(speed_val)

            # Annotate frame
            draw_calibration_lines(frame, line_a_y, line_b_y)
            draw_tracks(frame, tracks, speed_records, units=args.units)
            last_spd = speeds_seen[-1] if speeds_seen else None
            avg_spd = sum(speeds_seen) / len(speeds_seen) if speeds_seen else None
            draw_hud(frame, last_spd, avg_spd, len(logged_ids), units=args.units)

            if writer_out:
                writer_out.write(frame)

            cv2.imshow("speedcam", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            elif key == ord("r"):
                tracker.reset()
                estimator.reset()
                logged_ids.clear()
                speeds_seen.clear()
                print("[speedcam] Stats reset.")
            elif key == ord("s"):
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                fname = f"screenshot_{ts}.png"
                cv2.imwrite(fname, frame)
                print(f"[speedcam] Saved {fname}")

    finally:
        vs.release()
        csv_fh.close()
        if writer_out:
            writer_out.release()
        cv2.destroyAllWindows()
        print(f"\n[speedcam] Done. Detections logged to detections.csv")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="speedcam — vehicle speed detector",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--source", default="0",
                   help="Video source: webcam index, file path, or RTSP/HTTP URL")
    p.add_argument("--calibrate", action="store_true",
                   help="Force recalibration even if a saved calibration exists")
    p.add_argument("--units", choices=["mph", "kph"], default="mph",
                   help="Speed units for display and CSV output")
    p.add_argument("--record", action="store_true",
                   help="Write annotated video to output.mp4")
    p.add_argument("--conf", type=float, default=0.35,
                   help="YOLO detection confidence threshold")
    p.add_argument("--max-distance", type=float, default=80.0,
                   help="Max pixel distance for centroid tracker matching")
    p.add_argument("--max-missing", type=int, default=5,
                   help="Frames a track can disappear before being dropped")
    p.add_argument("--skip-frames", type=int, default=1,
                   help="Run YOLO inference every N frames (tracker runs every frame)")
    p.add_argument("--source-fps", type=float, default=None,
                   help="Override reported FPS for speed calculations (useful for RTSP)")
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
