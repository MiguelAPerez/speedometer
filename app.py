"""
app.py — Flask UI for speedcam.

Run with:  python app.py
Then open:  http://localhost:5000

Nothing is written outside this project folder.
"""

from __future__ import annotations

import csv
import io
import json
import os
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template_string, request, send_file

from speedcam.core import VideoSource, load_calibration, save_calibration, source_key, clear_calibration, is_live_camera
from speedcam.detector import Detector
from speedcam.overlay import draw_track, draw_hud, draw_tracks
from speedcam.speed import SpeedEstimator
from speedcam.tracker import CentroidTracker

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Global pipeline state (protected by _lock)
# ---------------------------------------------------------------------------

_lock = threading.Lock()

_state = {
    "running": False,
    "source": None,          # int or str
    "calibration": None,     # dict or None
    "units": "mph",
    "detections": [],        # list of dicts for CSV / table
    "speeds_seen": [],
    "logged_ids": set(),
    "last_frame": None,      # latest annotated BGR frame (numpy array)
    "preview_frame": None,   # first frame for calibration preview
    "tmp_video_path": None,  # path to uploaded video temp file
    "error": None,
}

_pipeline_thread: Optional[threading.Thread] = None


def _reset_stats():
    with _lock:
        _state["detections"] = []
        _state["speeds_seen"] = []
        _state["logged_ids"] = set()
        _state["error"] = None


# ---------------------------------------------------------------------------
# Pipeline thread
# ---------------------------------------------------------------------------

def _pipeline(source, calibration: dict, units: str):
    """Runs in a background thread — reads frames, detects, tracks, estimates speed."""
    try:
        detector = Detector(model_path="yolo11s.pt")
        tracker = CentroidTracker(graveyard_max_frames=150)
        estimator = SpeedEstimator.from_track(
            calibration["points"],
            calibration["distances"],
        )
        cal_points    = calibration["points"]
        cal_distances = calibration["distances"]

        try:
            vs = VideoSource(source)
        except Exception as e:
            with _lock:
                _state["running"] = False
                _state["error"] = str(e)
            return

        # Throttle playback to source FPS so video files don't race through
        # faster than the browser can display them.
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

                # Sleep for whatever time remains in this frame's budget
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


# ---------------------------------------------------------------------------
# MJPEG generator
# ---------------------------------------------------------------------------

def _mjpeg_stream():
    while True:
        with _lock:
            frame = _state["last_frame"]

        if frame is None:
            # Send a blank grey frame while waiting
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
        time.sleep(0.03)  # ~30 fps cap


def _preview_jpeg() -> bytes:
    """Return the preview frame as JPEG bytes, or a placeholder."""
    with _lock:
        frame = _state["preview_frame"]
        cal = _state["calibration"]

    if frame is None:
        blank = np.full((360, 640, 3), 40, dtype=np.uint8)
        cv2.putText(blank, "No preview — upload a file or check webcam",
                    (60, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
        frame = blank

    if cal:
        draw_calibration_lines(frame.copy(), cal.get("line_a_y"), cal.get("line_b_y"))

    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf.tobytes()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/video_feed")
def video_feed():
    return Response(_mjpeg_stream(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/preview_frame")
def preview_frame():
    return Response(_preview_jpeg(), mimetype="image/jpeg")


@app.route("/api/upload", methods=["POST"])
def upload():
    """Accept an uploaded video file, save to a temp path, grab the first frame."""
    f = request.files.get("video")
    if not f:
        return jsonify({"ok": False, "error": "No file"}), 400

    suffix = Path(f.filename).suffix or ".mp4"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix,
                                      dir=Path(__file__).parent / "tmp")
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

    # Grab the first frame for the calibration preview
    try:
        vs = VideoSource(tmp.name)
        frame = vs.grab_single_frame()
        vs.release()
        if frame is not None:
            with _lock:
                _state["preview_frame"] = frame.copy()
            return jsonify({"ok": True,
                            "w": frame.shape[1], "h": frame.shape[0]})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

    return jsonify({"ok": False, "error": "Could not read first frame"}), 500


@app.route("/api/grab_webcam_frame", methods=["POST"])
def grab_webcam_frame():
    """Grab a single frame from the webcam for calibration preview."""
    try:
        vs = VideoSource(0)
        frame = vs.grab_single_frame()
        vs.release()
        if frame is not None:
            with _lock:
                _state["preview_frame"] = frame.copy()
                _state["tmp_video_path"] = None
                _state["calibration"] = None
            return jsonify({"ok": True, "w": frame.shape[1], "h": frame.shape[0]})
        return jsonify({"ok": False, "error": "Empty frame"}), 500
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/calibrate", methods=["POST"])
def calibrate():
    """Save track calibration for the current source."""
    data = request.json
    try:
        points    = data["points"]     # [{x, y}, ...]
        distances = data["distances"]  # [float, ...]
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

    cal = {"points": points, "distances": distances,
           "frame_w": frame_w, "frame_h": frame_h}
    with _lock:
        _state["calibration"] = cal

    return jsonify({"ok": True})


@app.route("/api/load_calibration", methods=["POST"])
def load_cal():
    """Try to load saved calibration for the current source.

    For video files and streams the saved entry is cleared immediately —
    they always need a fresh calibration.
    """
    data = request.json or {}
    frame_w = int(data.get("frame_w", 0))
    frame_h = int(data.get("frame_h", 0))

    with _lock:
        src = _state["tmp_video_path"] if _state["tmp_video_path"] else 0

    # Files and streams: wipe any stale calibration and force recalibration.
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


@app.route("/api/start", methods=["POST"])
def start():
    global _pipeline_thread
    data = request.json or {}

    with _lock:
        if _state["running"]:
            return jsonify({"ok": False, "error": "Already running"}), 400
        cal = _state["calibration"]
        src = _state["tmp_video_path"] if _state["tmp_video_path"] else 0

    if not cal:
        return jsonify({"ok": False, "error": "No calibration set"}), 400

    units = data.get("units", "mph")
    _reset_stats()

    with _lock:
        _state["running"] = True
        _state["units"] = units
        _state["source"] = src

    _pipeline_thread = threading.Thread(
        target=_pipeline, args=(src, cal, units), daemon=True
    )
    _pipeline_thread.start()
    return jsonify({"ok": True})


@app.route("/api/stop", methods=["POST"])
def stop():
    with _lock:
        _state["running"] = False
    return jsonify({"ok": True})


@app.route("/api/reset", methods=["POST"])
def reset():
    with _lock:
        _state["running"] = False
    _reset_stats()
    return jsonify({"ok": True})


@app.route("/api/stats")
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
        "detections": detections[-50:],  # last 50 for the table
    })


@app.route("/api/download_csv")
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


# ---------------------------------------------------------------------------
# Inline HTML — single page, no external dependencies
# ---------------------------------------------------------------------------

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>speedcam</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  :root {
    --bg: #111;
    --surface: #1c1c1e;
    --border: #2c2c2e;
    --accent: #0a84ff;
    --green: #30d158;
    --red: #ff453a;
    --text: #f2f2f7;
    --muted: #8e8e93;
    --line-a: #00ffb4;
    --line-b: #ff8c00;
  }
  body { background: var(--bg); color: var(--text); font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; font-size: 14px; display: flex; height: 100vh; overflow: hidden; }

  /* Sidebar */
  #sidebar { width: 240px; min-width: 240px; background: var(--surface); border-right: 1px solid var(--border); display: flex; flex-direction: column; padding: 20px 16px; gap: 16px; overflow-y: auto; }
  #sidebar h1 { font-size: 18px; font-weight: 700; letter-spacing: -0.5px; }
  .section-label { font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.8px; color: var(--muted); margin-bottom: 6px; }
  .radio-group { display: flex; flex-direction: column; gap: 6px; }
  .radio-group label { display: flex; align-items: center; gap: 8px; cursor: pointer; }
  .btn { width: 100%; padding: 9px 12px; border: none; border-radius: 8px; font-size: 14px; font-weight: 600; cursor: pointer; transition: opacity .15s; }
  .btn:hover { opacity: .85; }
  .btn:disabled { opacity: .4; cursor: not-allowed; }
  .btn-primary { background: var(--accent); color: #fff; }
  .btn-danger  { background: var(--red);    color: #fff; }
  .btn-neutral { background: var(--border); color: var(--text); }
  .kpi-block { display: flex; flex-direction: column; gap: 10px; }
  .kpi { background: var(--bg); border-radius: 8px; padding: 10px 12px; }
  .kpi-label { font-size: 11px; color: var(--muted); margin-bottom: 2px; }
  .kpi-value { font-size: 22px; font-weight: 700; font-variant-numeric: tabular-nums; }
  #file-drop { border: 2px dashed var(--border); border-radius: 8px; padding: 14px; text-align: center; color: var(--muted); cursor: pointer; font-size: 12px; transition: border-color .2s; }
  #file-drop:hover, #file-drop.dragover { border-color: var(--accent); color: var(--accent); }
  #file-drop input { display: none; }
  select { width: 100%; background: var(--bg); color: var(--text); border: 1px solid var(--border); border-radius: 6px; padding: 7px 10px; font-size: 13px; }
  .dist-row { display: flex; align-items: center; gap: 8px; }
  .dist-row input[type=number] { flex: 1; background: var(--bg); color: var(--text); border: 1px solid var(--border); border-radius: 6px; padding: 7px 10px; font-size: 13px; }
  .dist-row span { color: var(--muted); font-size: 12px; white-space: nowrap; }
  #status-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--muted); display: inline-block; margin-right: 6px; }
  #status-dot.live { background: var(--green); box-shadow: 0 0 6px var(--green); animation: pulse 1.2s infinite; }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.4} }

  /* Main */
  #main { flex: 1; display: flex; flex-direction: column; overflow: hidden; }
  #tabs { display: flex; border-bottom: 1px solid var(--border); }
  .tab { padding: 12px 20px; cursor: pointer; border-bottom: 2px solid transparent; color: var(--muted); font-weight: 500; transition: color .15s; }
  .tab.active { color: var(--text); border-bottom-color: var(--accent); }
  .tab-panel { display: none; flex: 1; overflow: auto; padding: 20px; }
  .tab-panel.active { display: flex; flex-direction: column; gap: 16px; }

  /* Calibration canvas */
  #cal-wrap { position: relative; display: inline-block; max-width: 100%; cursor: crosshair; }
  #cal-img { display: block; max-width: 100%; border-radius: 8px; border: 1px solid var(--border); }
  #cal-canvas { position: absolute; top: 0; left: 0; width: 100%; height: 100%; border-radius: 8px; }
  .cal-hint { color: var(--muted); font-size: 13px; }
  .line-badge { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 4px; }
  .line-a { background: var(--line-a); }
  .line-b { background: var(--line-b); }

  /* Live feed */
  #feed-wrap { position: relative; max-width: 100%; }
  #live-img { display: block; max-width: 100%; border-radius: 8px; border: 1px solid var(--border); }

  /* Detections table */
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th { text-align: left; padding: 8px 12px; color: var(--muted); font-weight: 600; border-bottom: 1px solid var(--border); }
  td { padding: 8px 12px; border-bottom: 1px solid var(--border); font-variant-numeric: tabular-nums; }
  tr:last-child td { border-bottom: none; }
  .pill { display: inline-block; padding: 2px 8px; border-radius: 20px; font-size: 11px; font-weight: 600; background: var(--border); }
  .tag-info { background: #0a84ff22; color: var(--accent); }
  #toast { position: fixed; bottom: 24px; right: 24px; background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 12px 18px; font-size: 13px; opacity: 0; transition: opacity .3s; pointer-events: none; }
  #toast.show { opacity: 1; }
</style>
</head>
<body>

<!-- ===== SIDEBAR ===== -->
<aside id="sidebar">
  <h1>🚗 speedcam</h1>

  <div>
    <div class="section-label">Source</div>
    <div class="radio-group">
      <label><input type="radio" name="src" value="webcam" checked> Webcam</label>
      <label><input type="radio" name="src" value="file"> Video file</label>
    </div>
  </div>

  <div id="file-section" style="display:none">
    <div id="file-drop" onclick="document.getElementById('file-input').click()">
      <input type="file" id="file-input" accept=".mp4,.mov,.avi,.mkv">
      Drop video here or click
    </div>
    <div id="file-name" style="font-size:11px;color:var(--muted);margin-top:4px"></div>
  </div>

  <div>
    <div class="section-label">Units</div>
    <select id="units-select">
      <option value="mph">mph</option>
      <option value="kph">kph</option>
    </select>
  </div>

  <div style="display:flex;flex-direction:column;gap:8px">
    <button class="btn btn-primary" id="btn-start" disabled>▶ Start</button>
    <button class="btn btn-danger"  id="btn-stop"  disabled>■ Stop</button>
    <button class="btn btn-neutral" id="btn-reset">↺ Reset stats</button>
    <button class="btn btn-neutral" id="btn-csv"   disabled>↓ Download CSV</button>
  </div>

  <div>
    <div class="section-label">Status</div>
    <div><span id="status-dot"></span><span id="status-text">Idle</span></div>
    <div id="error-msg" style="color:var(--red);font-size:12px;margin-top:4px"></div>
  </div>

  <div class="kpi-block">
    <div class="section-label">Stats</div>
    <div class="kpi"><div class="kpi-label">Last speed</div><div class="kpi-value" id="kpi-last">—</div></div>
    <div class="kpi"><div class="kpi-label">Avg speed</div><div class="kpi-value" id="kpi-avg">—</div></div>
    <div class="kpi"><div class="kpi-label">Vehicles</div><div class="kpi-value" id="kpi-count">0</div></div>
  </div>
</aside>

<!-- ===== MAIN ===== -->
<main id="main">
  <div id="tabs">
    <div class="tab active" data-tab="calibrate">1 — Calibrate</div>
    <div class="tab" data-tab="live">2 — Live</div>
    <div class="tab" data-tab="detections">3 — Detections</div>
  </div>

  <!-- CALIBRATE TAB -->
  <div class="tab-panel active" id="tab-calibrate">
    <div style="display:flex;gap:12px;align-items:center;flex-wrap:wrap">
      <button class="btn btn-neutral" id="btn-grab-webcam" style="width:auto;padding:8px 14px">📷 Grab webcam frame</button>
      <span class="cal-hint">or upload a video file in the sidebar</span>
    </div>

    <div class="cal-hint" id="cal-instruction">Click on the road to place point <strong>A</strong>, then keep clicking to extend the track. Enter each segment's distance below.</div>

    <div id="cal-wrap">
      <img id="cal-img" src="/preview_frame" alt="preview">
      <canvas id="cal-canvas"></canvas>
    </div>

    <!-- Per-segment distance inputs injected here by JS -->
    <div id="seg-inputs" style="display:flex;flex-direction:column;gap:8px"></div>

    <div style="display:flex;gap:8px;margin-top:4px;flex-wrap:wrap">
      <button class="btn btn-primary" id="btn-save-cal" style="width:auto;padding:8px 16px" disabled>💾 Save track</button>
      <button class="btn btn-neutral" id="btn-undo-pt" style="width:auto;padding:8px 12px">↩ Undo last point</button>
      <button class="btn btn-neutral" id="btn-clear-cal" style="width:auto;padding:8px 12px">✕ Clear all</button>
    </div>
    <div id="cal-status" style="font-size:12px;color:var(--muted);margin-top:4px"></div>
  </div>

  <!-- LIVE TAB -->
  <div class="tab-panel" id="tab-live">
    <div id="feed-wrap">
      <img id="live-img" src="/video_feed" alt="live feed">
    </div>
  </div>

  <!-- DETECTIONS TAB -->
  <div class="tab-panel" id="tab-detections">
    <div id="table-wrap">
      <p style="color:var(--muted)">No detections yet.</p>
    </div>
  </div>
</main>

<div id="toast"></div>

<script>
// ── Tabs ──────────────────────────────────────────────────────────────────
document.querySelectorAll('.tab').forEach(t => {
  t.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach(x => x.classList.remove('active'));
    document.querySelectorAll('.tab-panel').forEach(x => x.classList.remove('active'));
    t.classList.add('active');
    document.getElementById('tab-' + t.dataset.tab).classList.add('active');
  });
});

// ── Source radio ──────────────────────────────────────────────────────────
document.querySelectorAll('input[name=src]').forEach(r => {
  r.addEventListener('change', () => {
    document.getElementById('file-section').style.display =
      r.value === 'file' ? 'block' : 'none';
  });
});

// ── File upload ───────────────────────────────────────────────────────────
const fileInput = document.getElementById('file-input');
const fileDrop  = document.getElementById('file-drop');
const fileName  = document.getElementById('file-name');

fileDrop.addEventListener('dragover', e => { e.preventDefault(); fileDrop.classList.add('dragover'); });
fileDrop.addEventListener('dragleave', () => fileDrop.classList.remove('dragover'));
fileDrop.addEventListener('drop', e => {
  e.preventDefault(); fileDrop.classList.remove('dragover');
  if (e.dataTransfer.files[0]) uploadFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener('change', () => { if (fileInput.files[0]) uploadFile(fileInput.files[0]); });

async function uploadFile(file) {
  fileName.textContent = 'Uploading ' + file.name + '…';
  const fd = new FormData(); fd.append('video', file);
  const r = await fetch('/api/upload', { method: 'POST', body: fd });
  const d = await r.json();
  if (d.ok) {
    fileName.textContent = file.name;
    frameSize = { w: d.w, h: d.h };
    refreshPreview();
    tryLoadCal();
    toast('File loaded');
  } else {
    fileName.textContent = '⚠ ' + d.error;
  }
}

// ── Calibration track drawer ──────────────────────────────────────────────
const calImg    = document.getElementById('cal-img');
const calCanvas = document.getElementById('cal-canvas');
const ctx       = calCanvas.getContext('2d');

let trackPts  = [];   // [{x, y}, ...]  pixel coords in natural image space
let frameSize = { w: 0, h: 0 };

const NODE_COLORS = ['#00ffb4','#ff8c00','#c850ff','#ffe066','#00cfff','#ff6b6b'];

calImg.addEventListener('load', () => {
  calCanvas.width  = calImg.naturalWidth;
  calCanvas.height = calImg.naturalHeight;
  redrawCal();
});

document.getElementById('btn-grab-webcam').addEventListener('click', async () => {
  const r = await fetch('/api/grab_webcam_frame', { method: 'POST' });
  const d = await r.json();
  if (d.ok) {
    frameSize = { w: d.w, h: d.h };
    refreshPreview();
    tryLoadCal();
  } else {
    toast('Webcam error: ' + d.error, true);
  }
});

calCanvas.addEventListener('click', e => {
  const rect = calCanvas.getBoundingClientRect();
  const sx = calCanvas.width  / rect.width;
  const sy = calCanvas.height / rect.height;
  trackPts.push({ x: (e.clientX - rect.left) * sx, y: (e.clientY - rect.top) * sy });
  redrawCal();
  rebuildSegInputs();
  updateCalState();
});

function redrawCal() {
  ctx.clearRect(0, 0, calCanvas.width, calCanvas.height);
  if (trackPts.length === 0) return;

  // Segment lines
  for (let i = 0; i < trackPts.length - 1; i++) {
    const p1 = trackPts[i], p2 = trackPts[i + 1];
    ctx.strokeStyle = '#888';
    ctx.lineWidth = 1.5;
    ctx.setLineDash([6, 4]);
    ctx.beginPath(); ctx.moveTo(p1.x, p1.y); ctx.lineTo(p2.x, p2.y); ctx.stroke();
    ctx.setLineDash([]);

    // Distance label at midpoint
    const mx = (p1.x + p2.x) / 2, my = (p1.y + p2.y) / 2;
    const dist = getSegDist(i);
    if (dist) {
      const lbl = dist + 'm';
      ctx.font = 'bold 13px -apple-system, sans-serif';
      const tw = ctx.measureText(lbl).width;
      ctx.fillStyle = 'rgba(30,30,30,0.75)';
      ctx.fillRect(mx - tw/2 - 3, my - 12, tw + 6, 18);
      ctx.fillStyle = '#ddd';
      ctx.fillText(lbl, mx - tw/2, my + 1);
    }
  }

  // Nodes
  const labels = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';
  trackPts.forEach((pt, i) => {
    const color = NODE_COLORS[i % NODE_COLORS.length];
    ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.arc(pt.x, pt.y, 10, 0, Math.PI * 2); ctx.stroke();
    ctx.beginPath(); ctx.arc(pt.x, pt.y, 3, 0, Math.PI * 2); ctx.fill();
    const lbl = labels[i] || String(i);
    const flip = pt.x + 40 > calCanvas.width;
    ctx.lineWidth = 2;
    ctx.beginPath();
    if (flip) { ctx.moveTo(pt.x - 22, pt.y); ctx.lineTo(pt.x - 10, pt.y); }
    else      { ctx.moveTo(pt.x + 10, pt.y); ctx.lineTo(pt.x + 22, pt.y); }
    ctx.stroke();
    ctx.font = 'bold 14px -apple-system, sans-serif';
    ctx.fillText(lbl, flip ? pt.x - 42 : pt.x + 26, pt.y + 5);
  });
}

function getSegDist(i) {
  const inp = document.getElementById('seg-dist-' + i);
  return inp ? inp.value : '';
}

function rebuildSegInputs() {
  const wrap = document.getElementById('seg-inputs');
  // Preserve existing values
  const existing = [];
  wrap.querySelectorAll('input').forEach(inp => existing.push(inp.value));
  wrap.innerHTML = '';
  const labels = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';
  for (let i = 0; i < trackPts.length - 1; i++) {
    const row = document.createElement('div');
    row.className = 'dist-row';
    row.style.cssText = 'display:flex;align-items:center;gap:8px;';
    const badge = document.createElement('span');
    badge.style.cssText = `display:inline-block;width:20px;height:20px;border-radius:50%;background:${NODE_COLORS[i % NODE_COLORS.length]};text-align:center;line-height:20px;font-size:11px;font-weight:700;color:#000;flex-shrink:0`;
    badge.textContent = labels[i] || i;
    const arrow = document.createElement('span');
    arrow.style.color = 'var(--muted)';
    arrow.textContent = '->';
    const badge2 = document.createElement('span');
    badge2.style.cssText = `display:inline-block;width:20px;height:20px;border-radius:50%;background:${NODE_COLORS[(i+1) % NODE_COLORS.length]};text-align:center;line-height:20px;font-size:11px;font-weight:700;color:#000;flex-shrink:0`;
    badge2.textContent = labels[i + 1] || (i + 1);
    const inp = document.createElement('input');
    inp.type = 'number'; inp.min = '0.1'; inp.step = '0.5'; inp.placeholder = 'metres';
    inp.id = 'seg-dist-' + i;
    inp.style.cssText = 'flex:1;background:var(--bg);color:var(--text);border:1px solid var(--border);border-radius:6px;padding:6px 10px;font-size:13px;';
    inp.value = existing[i] || '';
    inp.addEventListener('input', () => redrawCal());
    const unit = document.createElement('span');
    unit.style.color = 'var(--muted)'; unit.textContent = 'm';
    row.append(badge, arrow, badge2, inp, unit);
    wrap.appendChild(row);
  }
}

function updateCalState() {
  const instr  = document.getElementById('cal-instruction');
  const savBtn = document.getElementById('btn-save-cal');
  const n = trackPts.length;
  if (n === 0) {
    instr.textContent = 'Click on the road to place point A, then keep clicking to extend the track.';
    savBtn.disabled = true;
  } else if (n === 1) {
    instr.textContent = 'Click to place point B.';
    savBtn.disabled = true;
  } else {
    instr.textContent = `${n} points, ${n-1} segment${n>2?'s':''}. Fill in the distances below, then save.`;
    savBtn.disabled = false;
  }
}

document.getElementById('btn-undo-pt').addEventListener('click', () => {
  if (trackPts.length === 0) return;
  trackPts.pop();
  redrawCal();
  rebuildSegInputs();
  updateCalState();
});

document.getElementById('btn-clear-cal').addEventListener('click', () => {
  trackPts = [];
  document.getElementById('seg-inputs').innerHTML = '';
  document.getElementById('cal-status').textContent = '';
  document.getElementById('btn-save-cal').disabled = true;
  redrawCal();
  updateCalState();
});

document.getElementById('btn-save-cal').addEventListener('click', async () => {
  if (trackPts.length < 2 || !frameSize.w) return;
  const distances = [];
  for (let i = 0; i < trackPts.length - 1; i++) {
    const v = parseFloat(getSegDist(i));
    if (!v || v <= 0) { toast(`Enter distance for segment ${i + 1}`, true); return; }
    distances.push(v);
  }
  const body = { points: trackPts, distances, frame_w: frameSize.w, frame_h: frameSize.h };
  const r = await fetch('/api/calibrate', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  const d = await r.json();
  if (d.ok) {
    document.getElementById('cal-status').textContent =
      `Saved — ${trackPts.length} points, ${distances.length} segment(s): ${distances.map(d=>d+'m').join(', ')}`;
    document.getElementById('btn-start').disabled = false;
    toast('Track calibration saved');
  } else {
    toast('Save failed: ' + d.error, true);
  }
});

async function tryLoadCal() {
  if (!frameSize.w) return;
  const r = await fetch('/api/load_calibration', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ frame_w: frameSize.w, frame_h: frameSize.h }),
  });
  const d = await r.json();
  if (d.ok) {
    const cal = d.calibration;
    trackPts = cal.points;
    rebuildSegInputs();
    // Restore saved distances
    cal.distances.forEach((dist, i) => {
      const inp = document.getElementById('seg-dist-' + i);
      if (inp) inp.value = dist;
    });
    redrawCal();
    updateCalState();
    document.getElementById('cal-status').textContent =
      `Loaded — ${trackPts.length} points, segments: ${cal.distances.map(d=>d+'m').join(', ')}`;
    document.getElementById('btn-save-cal').disabled = false;
    document.getElementById('btn-start').disabled = false;
    toast('Saved track loaded');
  }
}

function refreshPreview() {
  calImg.src = '/preview_frame?' + Date.now();
}

// ── Controls ──────────────────────────────────────────────────────────────
document.getElementById('btn-start').addEventListener('click', async () => {
  const units = document.getElementById('units-select').value;
  const r = await fetch('/api/start', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ units }),
  });
  const d = await r.json();
  if (d.ok) {
    // Switch to live tab
    document.querySelectorAll('.tab').forEach(x => x.classList.remove('active'));
    document.querySelectorAll('.tab-panel').forEach(x => x.classList.remove('active'));
    document.querySelector('[data-tab=live]').classList.add('active');
    document.getElementById('tab-live').classList.add('active');
    document.getElementById('live-img').src = '/video_feed?' + Date.now();
  } else {
    toast(d.error, true);
  }
});

document.getElementById('btn-stop').addEventListener('click', async () => {
  await fetch('/api/stop', { method: 'POST' });
});

document.getElementById('btn-reset').addEventListener('click', async () => {
  await fetch('/api/reset', { method: 'POST' });
  document.getElementById('kpi-last').textContent = '—';
  document.getElementById('kpi-avg').textContent  = '—';
  document.getElementById('kpi-count').textContent = '0';
  document.getElementById('table-wrap').innerHTML = '<p style="color:var(--muted)">No detections yet.</p>';
  document.getElementById('btn-csv').disabled = true;
  toast('Stats reset');
});

document.getElementById('btn-csv').addEventListener('click', () => {
  window.location = '/api/download_csv';
});

// ── Stats polling ─────────────────────────────────────────────────────────
let lastCount = 0;

async function pollStats() {
  try {
    const r = await fetch('/api/stats');
    const d = await r.json();

    // Status indicator
    const dot  = document.getElementById('status-dot');
    const stxt = document.getElementById('status-text');
    const errEl = document.getElementById('error-msg');
    if (d.running) {
      dot.className = 'live'; stxt.textContent = 'Live';
      document.getElementById('btn-stop').disabled = false;
      document.getElementById('btn-start').disabled = true;
    } else {
      dot.className = ''; stxt.textContent = 'Idle';
      document.getElementById('btn-stop').disabled = true;
      document.getElementById('btn-start').disabled = false;
    }
    errEl.textContent = d.error || '';

    // KPIs
    if (d.last_speed !== null) {
      document.getElementById('kpi-last').textContent = d.last_speed + ' ' + d.units;
      document.getElementById('kpi-avg').textContent  = d.avg_speed  + ' ' + d.units;
    }
    document.getElementById('kpi-count').textContent = d.count;

    // CSV button
    document.getElementById('btn-csv').disabled = d.count === 0;

    // Detections table (only re-render when count changes)
    if (d.count !== lastCount) {
      lastCount = d.count;
      renderTable(d.detections, d.units);
    }
  } catch(e) { /* server not yet up */ }
}

function renderTable(rows, units) {
  if (!rows || rows.length === 0) {
    document.getElementById('table-wrap').innerHTML =
      '<p style="color:var(--muted)">No detections yet.</p>';
    return;
  }
  const speedCol = `speed_${units}`;
  let html = `<table><thead><tr>
    <th>Time</th><th>ID</th><th>Type</th><th>Speed (${units})</th><th>Dir</th>
  </tr></thead><tbody>`;
  [...rows].reverse().forEach(row => {
    html += `<tr>
      <td>${row.time}</td>
      <td><span class="pill">#${row.id}</span></td>
      <td><span class="pill tag-info">${row.type}</span></td>
      <td>${row[speedCol] ?? '—'}</td>
      <td>${row.direction}</td>
    </tr>`;
  });
  html += '</tbody></table>';
  document.getElementById('table-wrap').innerHTML = html;
}

setInterval(pollStats, 800);
pollStats();

// ── Toast ─────────────────────────────────────────────────────────────────
function toast(msg, isError=false) {
  const el = document.getElementById('toast');
  el.textContent = (isError ? '⚠ ' : '✓ ') + msg;
  el.style.borderColor = isError ? 'var(--red)' : 'var(--green)';
  el.classList.add('show');
  clearTimeout(el._t);
  el._t = setTimeout(() => el.classList.remove('show'), 2800);
}
</script>
</body>
</html>
"""

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="speedcam web UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    # Temp folder for uploaded videos — inside the project, not system temp
    Path("tmp").mkdir(exist_ok=True)

    print(f"speedcam running at http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)
