#!/usr/bin/env python3
"""
test_tracking.py — Quick tracking smoke-test.

Runs detection + tracker on a video without calibration.
Color codes tracks, prints a RESURRECTED banner when a graveyard
track comes back, and shows a summary at the end.

Usage:
    python3 test_tracking.py <video_path>
    python3 test_tracking.py <video_path> --model yolov8n.pt  # override model
    python3 test_tracking.py <video_path> --no-display        # headless, prints summary only
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from speedcam.detector import DEFAULT_MODEL
from speedcam.pipeline import build_detector, build_tracker


# Distinct BGR colors for up to 20 concurrent track IDs
_PALETTE = [
    (0, 200, 255), (0, 255, 100), (255, 80, 0),  (200, 0, 255), (0, 255, 220),
    (255, 200, 0), (0, 100, 255), (180, 255, 0),  (255, 0, 150), (0, 220, 180),
    (255, 140, 0), (0, 80, 220),  (220, 0, 80),   (80, 255, 0),  (0, 180, 255),
    (255, 60, 180),(60, 255, 180),(180, 60, 255),  (255, 180, 60),(60, 180, 255),
]

FONT = cv2.FONT_HERSHEY_SIMPLEX


def _color(track_id: int) -> tuple:
    return _PALETTE[track_id % len(_PALETTE)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("source", help="Path to video file")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help=f"YOLO model weights (default: {DEFAULT_MODEL})")
    ap.add_argument("--conf", type=float, default=0.40,
                    help="Detection confidence threshold")
    ap.add_argument("--max-missing", type=int, default=25,
                    help="Frames before a track moves to graveyard")
    ap.add_argument("--graveyard-frames", type=int, default=150,
                    help="Frames a graveyard track is held for ReID")
    ap.add_argument("--skip", type=int, default=1,
                    help="Run YOLO every N frames")
    ap.add_argument("--no-display", action="store_true",
                    help="Headless mode — skip OpenCV window")
    ap.add_argument("--scale", type=float, default=0.35,
                    help="Display scale factor (default 0.35 for 4K source)")
    args = ap.parse_args()

    source = args.source
    if not Path(source).exists():
        print(f"[ERROR] File not found: {source}")
        sys.exit(1)

    print(f"[test] Loading model: {args.model}")
    detector = build_detector(model_path=args.model, conf_threshold=args.conf)
    tracker = build_tracker(
        max_missing=args.max_missing,
        graveyard_max_frames=args.graveyard_frames,
    )

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("[ERROR] Could not open video.")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[test] {w}×{h} @ {fps:.1f}fps  |  {total_frames} frames  |  {total_frames/fps:.1f}s")
    print("[test] Press q to quit, space to pause\n")

    # Tracking bookkeeping
    prev_active: set[int] = set()
    ever_seen: set[int] = set()
    resurrections: list[dict] = []
    frame_idx = 0

    t0 = time.monotonic()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        # Detect every N frames
        if frame_idx % args.skip == 0:
            detections = detector.detect(frame)
        else:
            detections = []

        tracks = tracker.update(detections, frame=frame)

        # Detect resurrections: track IDs that are now active but were
        # previously in the graveyard (i.e., seen before, went missing, came back)
        current_active = set(tracks.keys())
        resurrected_this_frame = current_active & (ever_seen - prev_active)
        for tid in resurrected_this_frame:
            info = {
                "frame": frame_idx,
                "time": frame_idx / fps,
                "track_id": tid,
            }
            resurrections.append(info)
            print(f"  [RESURRECTED] track {tid} at frame {frame_idx} ({frame_idx/fps:.1f}s)")

        ever_seen |= current_active
        prev_active = current_active

        if not args.no_display:
            disp = frame.copy()

            # Draw bounding boxes and IDs
            for tid, track in tracks.items():
                x1, y1, x2, y2 = [int(v) for v in track.bbox]
                color = _color(tid)
                thickness = 3 if track.detected_this_frame else 1

                cv2.rectangle(disp, (x1, y1), (x2, y2), color, thickness)

                label = f"#{tid} {track.label}"
                if not track.detected_this_frame:
                    label += " [pred]"

                # Check if this was just resurrected
                if tid in resurrected_this_frame:
                    label += " *** RESURRECTED ***"
                    cv2.rectangle(disp, (x1-4, y1-4), (x2+4, y2+4), (0, 0, 255), 3)

                (tw, th), _ = cv2.getTextSize(label, FONT, 0.65, 2)
                cv2.rectangle(disp, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
                cv2.putText(disp, label, (x1 + 2, y1 - 4), FONT, 0.65, (0, 0, 0), 2, cv2.LINE_AA)

            # HUD
            elapsed = time.monotonic() - t0
            proc_fps = frame_idx / elapsed if elapsed > 0 else 0
            gyard_count = len(tracker._graveyard)
            hud = [
                f"Frame {frame_idx}/{total_frames}",
                f"Active tracks: {len(tracks)}   Graveyard: {gyard_count}",
                f"Resurrections so far: {len(resurrections)}",
                f"Processing: {proc_fps:.1f} fps",
            ]
            y_off = 32
            for line in hud:
                cv2.putText(disp, line, (12, y_off), FONT, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(disp, line, (12, y_off), FONT, 0.7, (0, 255, 200), 2, cv2.LINE_AA)
                y_off += 30

            # Scale down for display
            dw = int(w * args.scale)
            dh = int(h * args.scale)
            disp = cv2.resize(disp, (dw, dh))
            cv2.imshow("tracking test", disp)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord(" "):
                cv2.waitKey(0)   # pause until any key

    cap.release()
    if not args.no_display:
        cv2.destroyAllWindows()

    # Summary
    elapsed = time.monotonic() - t0
    print(f"\n{'='*50}")
    print(f"Processed {frame_idx} frames in {elapsed:.1f}s ({frame_idx/elapsed:.1f} fps)")
    print(f"Unique track IDs assigned: {len(ever_seen)}  ({sorted(ever_seen)})")
    print(f"Resurrections: {len(resurrections)}")
    for r in resurrections:
        print(f"  track {r['track_id']} resurrected at frame {r['frame']} ({r['time']:.1f}s)")
    print(f"Remaining graveyard entries: {len(tracker._graveyard)}")
    print("="*50)


if __name__ == "__main__":
    main()
