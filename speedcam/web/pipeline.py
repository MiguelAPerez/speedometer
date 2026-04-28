from __future__ import annotations

import time
from datetime import datetime

import cv2
import numpy as np

from speedcam.core import VideoSource
from speedcam.overlay import draw_track, draw_hud, draw_tracks
from speedcam.pipeline import build_detector, build_tracker
from speedcam.speed import SpeedEstimator

from .state import _lock, _state


def run_pipeline(source, calibration: dict, units: str) -> None:
    """Runs in a background thread — reads frames, detects, tracks, estimates speed."""
    try:
        detector = build_detector()
        tracker = build_tracker()
        estimator = SpeedEstimator.from_track(
            calibration["points"],
            calibration["distances"],
        )
        cal_points = calibration["points"]
        cal_distances = calibration["distances"]

        try:
            vs = VideoSource(source)
        except Exception as e:
            with _lock:
                _state["running"] = False
                _state["error"] = str(e)
            return

        fps = vs.fps or 30.0
        frame_interval = 1.0 / fps

        try:
            while True:
                t_frame_start = time.monotonic()

                with _lock:
                    if not _state["running"]:
                        break

                ok, frame = vs.read()
                if not ok:
                    break

                dets = detector.detect(frame)
                tracks = tracker.update(dets, frame=frame)
                records = estimator.update(tracks, frame_ts=t_frame_start)

                with _lock:
                    for tid, rec in records.items():
                        if tid not in _state["logged_ids"]:
                            spd = rec.speed_mph if units == "mph" else rec.speed_kph
                            _state["speeds_seen"].append(spd)
                            _state["logged_ids"].add(tid)
                            trk = tracks.get(tid)
                            _state["detections"].append({
                                "time": datetime.now().strftime("%H:%M:%S"),
                                "id": tid,
                                "type": trk.label if trk else "vehicle",
                                f"speed_{units}": round(spd, 1),
                                "direction": rec.direction,
                            })

                    spds = _state["speeds_seen"]
                    last_spd = spds[-1] if spds else None
                    avg_spd = sum(spds) / len(spds) if spds else None
                    count = len(_state["logged_ids"])

                draw_track(frame, cal_points, cal_distances)
                draw_tracks(frame, tracks, records, units=units)
                draw_hud(frame, last_spd, avg_spd, count, units=units)

                with _lock:
                    _state["last_frame"] = frame.copy()

                elapsed = time.monotonic() - t_frame_start
                sleep_t = frame_interval - elapsed
                if sleep_t > 0:
                    time.sleep(sleep_t)

        finally:
            vs.release()
            with _lock:
                _state["running"] = False

    except Exception as e:
        with _lock:
            _state["running"] = False
            _state["error"] = f"Pipeline error: {e}"


def mjpeg_stream():
    while True:
        with _lock:
            frame = _state["last_frame"]

        if frame is None:
            blank = np.full((360, 640, 3), 40, dtype=np.uint8)
            cv2.putText(blank, "Waiting for feed...", (160, 190),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 180), 2)
            frame = blank

        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if ok:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
            )
        time.sleep(0.03)


def preview_jpeg() -> bytes:
    with _lock:
        frame = _state["preview_frame"]

    if frame is None:
        blank = np.full((360, 640, 3), 40, dtype=np.uint8)
        cv2.putText(blank, "No preview — upload a file or check webcam",
                    (60, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
        frame = blank

    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf.tobytes()
