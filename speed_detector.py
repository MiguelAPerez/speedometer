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
import select
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from speedcam.core import VideoSource, load_calibration, save_calibration, clear_calibration, is_live_camera
from speedcam.detector import Detector
from speedcam.tracker import CentroidTracker
from speedcam.speed import SpeedEstimator, SpeedRecord
from speedcam.overlay import draw_tracks, draw_track, draw_hud


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

CSV_COLUMNS = [
    "timestamp", "track_id", "vehicle_type",
    "speed_mph", "speed_kph", "direction", "sample_count",
]


def _open_csv(path: str):
    """Open (or append to) a CSV file and return (file_handle, writer)."""
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
# Calibration UI (OpenCV window + terminal prompts)
# ---------------------------------------------------------------------------

def run_calibration_ui(frame: np.ndarray, source) -> Optional[dict]:
    """
    Multi-segment track calibration UI.

    Click nodes A, B, C … on the OpenCV preview window.  After each new node
    (beyond the first) you'll be prompted in the terminal to enter the
    real-world distance for that segment.  Blank input undoes the last node.
    When you have ≥ 2 nodes, press Enter (empty line) in the terminal to
    finish.  Press ESC in the window to cancel at any time.

    Returns {points, distances, frame_w, frame_h} or None on cancel.
    """
    clone = frame.copy()
    h, w = clone.shape[:2]
    pts: List[dict] = []        # {"x": float, "y": float}
    distances: List[float] = []
    done = [False]
    cancelled = [False]
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    WIN = "Calibration — speedcam"

    def _redraw():
        img = clone.copy()
        draw_track(img, pts, distances)
        n = len(pts)
        if n == 0:
            msg = "Click to place node A"
        elif n == 1:
            msg = "Click node B — then enter distance in terminal"
        else:
            msg = f"{n} nodes placed  |  click for more  |  blank Enter = finish"
        cv2.putText(img, msg, (10, 30), FONT, 0.60, (0, 255, 200), 2, cv2.LINE_AA)
        cv2.putText(img, "ESC = cancel", (10, h - 12), FONT, 0.46, (180, 180, 180), 1, cv2.LINE_AA)
        cv2.imshow(WIN, img)

    def _mouse_cb(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN or done[0] or cancelled[0]:
            return
        pts.append({"x": float(x), "y": float(y)})
        _redraw()
        if len(pts) < 2:
            return
        seg = len(pts) - 1
        lp = chr(64 + seg)   # A, B, C …
        lc = chr(65 + seg)   # B, C, D …
        print(f"\n[Cal] Segment {lp}→{lc} placed.")
        while True:
            raw = input(f"  Distance {lp}→{lc} in metres (blank = undo node): ").strip()
            if raw == "":
                pts.pop()
                _redraw()
                print("  Node removed.")
                return
            try:
                d = float(raw)
                if d <= 0:
                    raise ValueError
                distances.append(d)
                print(f"  Saved {d} m.  Click next node or press Enter (blank) to finish.")
                _redraw()
                return
            except ValueError:
                print("  Please enter a positive number (e.g. 8.5)")

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WIN, _mouse_cb)
    _redraw()

    print("\n[Calibration] Multi-segment track drawer")
    print("  Click nodes A, B, C … on the window.")
    print("  Enter each segment's real-world length when prompted.")
    print("  Press Enter (blank) in the terminal when finished (≥ 2 nodes).")
    print("  Press ESC in the window to cancel.\n")

    while not done[0] and not cancelled[0]:
        cv2.setWindowTitle(WIN, f"Calibration — {len(pts)} node(s) placed")
        key = cv2.waitKey(150) & 0xFF
        if key == 27:
            cancelled[0] = True
            break

        # Non-blocking check for an empty Enter on stdin (the "done" signal)
        r, _, _ = select.select([sys.stdin], [], [], 0)
        if r:
            line = sys.stdin.readline().strip()
            if line == "" and len(pts) >= 2:
                done[0] = True

    cv2.destroyWindow(WIN)

    if cancelled[0] or len(pts) < 2:
        if len(pts) < 2 and not cancelled[0]:
            print("[Cal] Need at least 2 nodes — calibration cancelled.")
        return None

    save_calibration(source, points=pts, distances=distances, frame_w=w, frame_h=h)
    labels = "".join(chr(65 + i) for i in range(len(pts)))
    print(f"[Calibration] Saved track {labels} ({len(distances)} segment(s)).\n")
    return {"points": pts, "distances": distances, "frame_w": w, "frame_h": h}


# ---------------------------------------------------------------------------
# Main processing loop
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    source = args.source
    if isinstance(source, str) and source.isdigit():
        source = int(source)

    print(f"[speedcam] Opening source: {source!r}")
    vs = VideoSource(source, source_fps_override=args.source_fps)
    w, h = vs.width, vs.height
    print(f"[speedcam] Frame size: {w}×{h}  FPS: {vs.fps:.1f}")

    # ---- Calibration ----
    # For video files and streams, always start fresh — clear any stale entry.
    if not is_live_camera(source):
        clear_calibration(source)

    calib = None
    if not args.calibrate and is_live_camera(source):
        calib = load_calibration(source, w, h)
        if calib:
            print("[speedcam] Loaded saved calibration for webcam.")

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

    cal_points   = calib["points"]
    cal_distances = calib["distances"]

    # ---- Pipeline ----
    detector  = Detector(conf_threshold=args.conf)
    tracker   = CentroidTracker(max_distance=args.max_distance, max_missing=args.max_missing)
    estimator = SpeedEstimator.from_track(cal_points, cal_distances)

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

            if frame_idx % skip == 0:
                detections = detector.detect(frame)
            else:
                detections = []

            tracks       = tracker.update(detections)
            speed_records = estimator.update(tracks, frame_ts=frame_ts)

            # Log first speed reading per track to CSV
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
            draw_track(frame, cal_points, cal_distances)
            draw_tracks(frame, tracks, speed_records, units=args.units)
            last_spd = speeds_seen[-1] if speeds_seen else None
            avg_spd  = sum(speeds_seen) / len(speeds_seen) if speeds_seen else None
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
    p.add_argument("--max-missing", type=int, default=25,
                   help="Frames a track can disappear before being dropped")
    p.add_argument("--skip-frames", type=int, default=1,
                   help="Run YOLO inference every N frames (tracker runs every frame)")
    p.add_argument("--source-fps", type=float, default=None,
                   help="Override reported FPS for speed calculations (useful for RTSP)")
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
