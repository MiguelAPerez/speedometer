from __future__ import annotations

import base64
import queue
import threading
import time
from datetime import datetime

import cv2
import numpy as np

from speedcam.core import VideoSource, is_live_camera
from speedcam.overlay import draw_track, draw_hud, draw_tracks
from speedcam.pipeline import build_detector, build_tracker
from speedcam.speed import SpeedEstimator

from .state import _lock, _state


def _crop_thumbnail(frame: np.ndarray, track) -> str | None:
    """Return a base64 JPEG data-URL of the vehicle's bounding box crop, or None."""
    try:
        x1, y1, x2, y2 = (int(v) for v in track.bbox)
        h, w = frame.shape[:2]
        pad_x = max(1, int((x2 - x1) * 0.10))
        pad_y = max(1, int((y2 - y1) * 0.10))
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w, x2 + pad_x)
        y2 = min(h, y2 + pad_y)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        # Resize to max 160×120 maintaining aspect ratio
        ch, cw = crop.shape[:2]
        scale = min(160 / cw, 120 / ch, 1.0)
        if scale < 1.0:
            crop = cv2.resize(crop, (int(cw * scale), int(ch * scale)), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 55])
        if not ok:
            return None
        b64 = base64.b64encode(buf.tobytes()).decode("ascii")
        return f"data:image/jpeg;base64,{b64}"
    except Exception:
        return None


def _flush_pending_tracks(units: str) -> None:
    """Log any tracks still in peak_speeds (e.g. car visible at end of video)."""
    for tid, p in list(_state["peak_speeds"].items()):
        if tid not in _state["logged_ids"]:
            _state["logged_ids"].add(tid)
            spds = p["speeds"]
            avg = round(sum(spds) / len(spds), 1) if spds else p["top"]
            _state["detections"].append({
                "time": p["time"],
                "id": tid,
                "type": p["type"],
                f"top_speed_{units}": round(p["top"], 1),
                f"avg_speed_{units}": avg,
                "direction": p["direction"],
                "thumbnail": p["thumbnail"],
            })
    _state["peak_speeds"].clear()


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

        # Reader thread decodes the next frame while inference runs on the current one.
        # Queue size 4 = ~133ms of buffer at 30fps.
        # Live cameras: drop on full to stay real-time (vs.read blocks at camera fps).
        # Video files: block on full so the reader can't race through the file faster
        # than inference consumes it — otherwise nearly all frames get dropped.
        frame_q: queue.Queue = queue.Queue(maxsize=4)
        _live = is_live_camera(source)

        def _reader() -> None:
            try:
                while True:
                    with _lock:
                        if not _state["running"]:
                            break
                    ok, frame = vs.read()
                    if not ok:
                        break
                    if _live:
                        try:
                            frame_q.put_nowait(frame)
                        except queue.Full:
                            pass
                    else:
                        frame_q.put(frame)
            finally:
                try:
                    frame_q.put(None, timeout=1)
                except queue.Full:
                    pass

        reader_t = threading.Thread(target=_reader, daemon=True)
        reader_t.start()

        try:
            while True:
                t_frame_start = time.monotonic()

                with _lock:
                    if not _state["running"]:
                        break

                try:
                    frame = frame_q.get(timeout=2.0)
                except queue.Empty:
                    break
                if frame is None:
                    break

                dets = detector.detect(frame)
                tracks = tracker.update(dets, frame=frame)
                records = estimator.update(tracks, frame_ts=t_frame_start)
                # Show last-known speed for tracks still warming up (< min_samples)
                # so resurrected or newly appeared cars don't flash "car" label
                for tid in tracks:
                    if tid not in records:
                        prior = estimator.get_record(tid)
                        if prior:
                            records[tid] = prior

                with _lock:
                    frame_max_spd = 0.0
                    for tid, rec in records.items():
                        spd = rec.speed_mph if units == "mph" else rec.speed_kph

                        # Track peak speed per vehicle; defer logging until
                        # the track leaves the scene so we record the real speed
                        # rather than the low warmup value from the first few frames.
                        if spd > 0 and tid not in _state["logged_ids"]:
                            trk = tracks.get(tid)
                            existing = _state["peak_speeds"].get(tid)
                            if existing is None:
                                thumb = _crop_thumbnail(frame, trk) if trk else None
                                _state["peak_speeds"][tid] = {
                                    "top": spd,
                                    "speeds": [spd],
                                    "time": datetime.now().strftime("%H:%M:%S"),
                                    "type": trk.label if trk else "vehicle",
                                    "direction": rec.direction,
                                    "thumbnail": thumb,
                                }
                            else:
                                existing["speeds"].append(spd)
                                if spd > existing["top"]:
                                    existing["top"] = spd
                                    existing["direction"] = rec.direction
                                    thumb = _crop_thumbnail(frame, trk) if trk else None
                                    if thumb:
                                        existing["thumbnail"] = thumb

                        if spd > frame_max_spd:
                            frame_max_spd = spd

                    # Flush tracks that just left the scene (in peak_speeds but
                    # no longer in active tracks) — log with their peak speed.
                    disappeared = [
                        tid for tid in list(_state["peak_speeds"])
                        if tid not in tracks and tid not in _state["logged_ids"]
                    ]
                    for tid in disappeared:
                        _state["logged_ids"].add(tid)
                        p = _state["peak_speeds"].pop(tid)
                        spds = p["speeds"]
                        avg = round(sum(spds) / len(spds), 1) if spds else p["top"]
                        _state["detections"].append({
                            "time": p["time"],
                            "id": tid,
                            "type": p["type"],
                            f"top_speed_{units}": round(p["top"], 1),
                            f"avg_speed_{units}": avg,
                            "direction": p["direction"],
                            "thumbnail": p["thumbnail"],
                        })

                    # Keep speeds_seen as a rolling buffer of per-frame max speeds
                    # so Last/Avg KPIs always reflect the current state.
                    if frame_max_spd > 0:
                        _state["speeds_seen"].append(frame_max_spd)
                        if len(_state["speeds_seen"]) > 200:
                            _state["speeds_seen"] = _state["speeds_seen"][-200:]

                    spds = _state["speeds_seen"]
                    last_spd = spds[-1] if spds else None
                    avg_spd = sum(spds) / len(spds) if spds else None
                    # Include pending (not-yet-logged) tracks in the count so the
                    # vehicles KPI ticks up while a car is still on screen.
                    count = len(_state["logged_ids"]) + len(_state["peak_speeds"])

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
            reader_t.join(timeout=2.0)
            with _lock:
                _flush_pending_tracks(units)
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
