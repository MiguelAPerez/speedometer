# speedcam

Vehicle speed detection using YOLOv8-nano + centroid tracking.  
Run as a CLI tool or open the Streamlit web UI.

---

## Setup

```bash
pip install -r requirements.txt
```

The first run will automatically download the YOLOv8-nano weights (~6 MB).

---

## CLI usage

```bash
# Webcam (default), mph
python speed_detector.py --source 0

# Video file, km/h, force new calibration
python speed_detector.py --source clip.mp4 --units kph --calibrate

# RTSP stream, record annotated output
python speed_detector.py --source rtsp://192.168.1.10:8554/cam --record

# Run inference every 2nd frame (faster on slower hardware)
python speed_detector.py --source 0 --skip-frames 2
```

**Keyboard shortcuts (live window)**

| Key | Action |
|-----|--------|
| `q` | Quit |
| `r` | Reset stats (keeps calibration) |
| `s` | Save screenshot |

**All flags**

| Flag | Default | Description |
|------|---------|-------------|
| `--source` | `0` | Webcam index, file path, or RTSP/HTTP URL |
| `--calibrate` | off | Force recalibration |
| `--units` | `mph` | `mph` or `kph` |
| `--record` | off | Write annotated video to `output.mp4` |
| `--conf` | `0.35` | YOLO detection confidence threshold |
| `--max-distance` | `80` | Max pixel distance for tracker matching |
| `--max-missing` | `5` | Frames before a track is dropped |
| `--skip-frames` | `1` | Run YOLO every N frames |
| `--source-fps` | auto | Override reported FPS (useful for RTSP) |

Detections are appended to `detections.csv` with columns:  
`timestamp, track_id, vehicle_type, speed_mph, speed_kph, direction, sample_count`

---

## Web UI (Flask)

```bash
python app.py
```

Then open `http://localhost:5000` in your browser.

Nothing is written outside the project folder — no global config, no telemetry.

---

## Physical calibration

The two reference lines are a **scale ruler** painted onto the scene — they tell the software how many pixels equal one metre.

1. **Pick two reference points** in the road that are a known distance apart and roughly parallel to the camera's horizontal axis. Painted road markings, lamp-post spacing, or chalk marks work well. A span of 8–15 m gives good resolution.

2. **Point the camera** so both reference points are clearly visible in the frame. Mount it above and to the side of the road so vehicles pass roughly perpendicular to the camera.

3. **Open the calibration UI** (CLI: `--calibrate` flag; Streamlit: Step 1). Click the frame at the level of reference point A, then at the level of reference point B. Enter the real-world distance between them.

4. **Calibration is saved** to `calibration.json` keyed by source (webcam index, file path, or URL). It's reused automatically on subsequent runs — you only need to recalibrate if you move the camera.

**Tips for good accuracy**

- Camera angle: mount as high as practical (2–5 m), angled slightly down. Steep angles compress the depth axis and reduce accuracy.
- Avoid heavy backlight — overexposed frames reduce detection confidence.
- For vehicles travelling at 30–60 mph in a typical residential or arterial setting, ±2–3 mph accuracy is realistic when the road is roughly perpendicular to the camera axis.
- Speed is computed from frame-to-frame centroid displacement, so a consistent frame rate matters. Use `--skip-frames 1` (default) on hardware that can keep up, or increase to 2–3 on slower machines.
- The plausibility filter in `speed.py` discards single-frame speed bursts — a rolling window of 3–15 frames is averaged per vehicle.

---

## Project structure

```
speedcam/
├── __init__.py
├── core.py        # VideoSource, calibration I/O
├── detector.py    # YOLOv8-nano wrapper
├── tracker.py     # Centroid nearest-neighbour tracker
├── speed.py       # Per-frame displacement speed estimator
└── overlay.py     # Frame annotation helpers
speed_detector.py  # CLI entry point
app.py             # Streamlit UI
requirements.txt
calibration.json   # Written at runtime
detections.csv     # Appended at runtime
```
