from __future__ import annotations

import csv
import io
import os
import tempfile
import threading
from pathlib import Path

from flask import Blueprint, Response, jsonify, render_template, request

from speedcam.core import VideoSource, load_calibration, save_calibration, clear_calibration, is_live_camera, get_data_dir

from .pipeline import run_pipeline, mjpeg_stream, preview_jpeg
from .state import _lock, _state, reset_stats

bp = Blueprint("main", __name__)


@bp.route("/")
def index():
    return render_template("index.html")


@bp.route("/video_feed")
def video_feed():
    return Response(mjpeg_stream(), mimetype="multipart/x-mixed-replace; boundary=frame")


@bp.route("/preview_frame")
def preview_frame():
    return Response(preview_jpeg(), mimetype="image/jpeg")


@bp.route("/api/upload", methods=["POST"])
def upload():
    f = request.files.get("video")
    if not f:
        return jsonify({"ok": False, "error": "No file"}), 400

    suffix = Path(f.filename).suffix or ".mp4"
    tmp = tempfile.NamedTemporaryFile(
        delete=False, suffix=suffix,
        dir=get_data_dir() / "tmp",
    )
    tmp.write(f.read())
    tmp.close()

    with _lock:
        old = _state.get("tmp_video_path")
        _state["tmp_video_path"] = tmp.name
        _state["calibration"] = None
        _state["preview_frame"] = None

    if old and os.path.exists(old):
        try:
            os.unlink(old)
        except OSError:
            pass

    try:
        vs = VideoSource(tmp.name)
        frame = vs.grab_single_frame()
        vs.release()
        if frame is not None:
            with _lock:
                _state["preview_frame"] = frame.copy()
            return jsonify({"ok": True, "w": frame.shape[1], "h": frame.shape[0]})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

    return jsonify({"ok": False, "error": "Could not read first frame"}), 500


@bp.route("/api/connect_source", methods=["POST"])
def connect_source():
    data = request.json or {}
    source = data.get("source", 0)
    if isinstance(source, str) and source.isdigit():
        source = int(source)

    try:
        vs = VideoSource(source)
        frame = vs.grab_single_frame()
        vs.release()
        if frame is not None:
            with _lock:
                _state["preview_frame"] = frame.copy()
                _state["tmp_video_path"] = source if not isinstance(source, int) else None
                _state["calibration"] = None
            return jsonify({"ok": True, "w": frame.shape[1], "h": frame.shape[0]})
        return jsonify({"ok": False, "error": "Could not grab frame from source"}), 500
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@bp.route("/api/grab_webcam_frame", methods=["POST"])
def grab_webcam_frame():
    # Keep for compatibility, but internally uses connect_source logic
    return connect_source()


@bp.route("/api/calibrate", methods=["POST"])
def calibrate():
    data = request.json
    try:
        points    = data["points"]
        distances = data["distances"]
        frame_w   = int(data["frame_w"])
        frame_h   = int(data["frame_h"])
        if len(points) < 2 or len(distances) != len(points) - 1:
            raise ValueError("Need at least 2 points and one distance per segment")
        distances = [float(d) for d in distances]
    except (KeyError, TypeError, ValueError) as e:
        return jsonify({"ok": False, "error": str(e)}), 400

    with _lock:
        src = _state["tmp_video_path"] if _state["tmp_video_path"] else 0

    save_calibration(src, points, distances, frame_w, frame_h)

    cal = {"points": points, "distances": distances, "frame_w": frame_w, "frame_h": frame_h}
    with _lock:
        _state["calibration"] = cal

    return jsonify({"ok": True})


@bp.route("/api/load_calibration", methods=["POST"])
def load_cal():
    data = request.json or {}
    frame_w = int(data.get("frame_w", 0))
    frame_h = int(data.get("frame_h", 0))

    with _lock:
        src = _state["tmp_video_path"] if _state["tmp_video_path"] else 0

    if not is_live_camera(src):
        clear_calibration(src)
        with _lock:
            _state["calibration"] = None
        return jsonify({"ok": False})

    if frame_w and frame_h:
        cal = load_calibration(src, frame_w, frame_h)
        if cal:
            with _lock:
                _state["calibration"] = cal
            return jsonify({"ok": True, "calibration": cal})

    return jsonify({"ok": False})


@bp.route("/api/start", methods=["POST"])
def start():
    data = request.json or {}

    with _lock:
        if _state["running"]:
            return jsonify({"ok": False, "error": "Already running"}), 400
        cal = _state["calibration"]
        src = _state["tmp_video_path"] if _state["tmp_video_path"] else 0

    if not cal:
        return jsonify({"ok": False, "error": "No calibration set"}), 400

    units = data.get("units", "mph")
    reset_stats()

    with _lock:
        _state["running"] = True
        _state["units"] = units
        _state["source"] = src

    t = threading.Thread(target=run_pipeline, args=(src, cal, units), daemon=True)
    with _lock:
        _state["pipeline_thread"] = t
    t.start()

    return jsonify({"ok": True})


@bp.route("/api/stop", methods=["POST"])
def stop():
    with _lock:
        _state["running"] = False
    return jsonify({"ok": True})


@bp.route("/api/reset", methods=["POST"])
def reset():
    with _lock:
        _state["running"] = False
    reset_stats()
    return jsonify({"ok": True})


@bp.route("/api/stats")
def stats():
    with _lock:
        spds = _state["speeds_seen"]
        units = _state["units"]
        running = _state["running"]
        error = _state["error"]
        detections = list(_state["detections"])

    last = round(spds[-1], 1) if spds else None
    avg = round(sum(spds) / len(spds), 1) if spds else None

    return jsonify({
        "running": running,
        "units": units,
        "last_speed": last,
        "avg_speed": avg,
        "count": len(detections),
        "error": error,
        "detections": detections[-50:],
    })


@bp.route("/api/map_segment")
def get_map_segment():
    with _lock:
        return jsonify({"ok": True, "segment": _state.get("map_segment"), "location": _state.get("map_location")})


@bp.route("/api/map_segment", methods=["POST"])
def set_map_segment():
    data = request.json or {}
    points = data.get("points", [])
    label  = str(data.get("label", ""))
    for p in points:
        if "lat" not in p or "lng" not in p:
            return jsonify({"ok": False, "error": "each point needs lat and lng"}), 400
        try:
            p["lat"] = float(p["lat"]); p["lng"] = float(p["lng"])
        except (TypeError, ValueError):
            return jsonify({"ok": False, "error": "lat/lng must be numbers"}), 400
    loc = data.get("location")
    with _lock:
        _state["map_segment"]  = {"points": points, "label": label} if points else None
        if loc:
            _state["map_location"] = loc
    return jsonify({"ok": True})


@bp.route("/api/download_csv")
def download_csv():
    with _lock:
        detections = list(_state["detections"])
        units = _state["units"]

    if not detections:
        return jsonify({"error": "No detections yet"}), 404

    buf = io.StringIO()
    cols = ["time", "id", "type", f"speed_{units}", "direction"]
    writer = csv.DictWriter(buf, fieldnames=cols, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(detections)

    return Response(
        buf.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=detections.csv"},
    )
