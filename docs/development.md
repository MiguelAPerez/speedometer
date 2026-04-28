# Development Guide

Everything you need to run Speedometer from source, understand the architecture, and contribute.

---

## Running from source

**Requirements:** Python 3.11+

```bash
git clone https://github.com/MiguelAPerez/speedometer
cd speedometer
python -m venv .env
source .env/bin/activate      # Windows: .env\Scripts\activate
pip install -r requirements.txt
python app.py
```

Open `http://localhost:8080`.

The first run downloads the YOLO model weights (~18 MB) automatically.

---

## Project structure

```
speedcam/
├── core.py        # VideoSource abstraction, calibration JSON I/O
├── detector.py    # YOLO wrapper — returns Detection objects per frame
├── tracker.py     # Kalman-filter centroid tracker with graveyard ReID
├── speed.py       # Per-frame displacement → speed (mph/kph)
├── overlay.py     # Frame annotation helpers
├── reid.py        # HSV histogram similarity for re-identification
├── pipeline.py    # Offline (CLI) processing pipeline
└── web/
    ├── pipeline.py # Streaming pipeline for the web UI
    ├── routes.py   # Flask Blueprint — all API endpoints
    └── state.py    # Shared in-memory state (thread-safe)

app.py             # Flask entry point (dev/server use)
launcher.py        # Packaged desktop entry point
speed_detector.py  # CLI entry point
speedometer.spec   # PyInstaller build spec
build/
└── installer.nsi  # Windows NSIS installer script
tests/
├── conftest.py
├── test_core.py
├── test_routes.py
└── test_speed.py
.github/workflows/
├── ci.yml         # pytest on push/PR
└── release.yml    # build Mac + Windows + Linux + Docker on tag
```

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

Detections are appended to `detections.csv`:  
`timestamp, track_id, vehicle_type, speed_mph, speed_kph, direction, sample_count`

---

## Architecture

### Detection pipeline

```
VideoSource → Detector (YOLO) → CentroidTracker → SpeedEstimator → Overlay
```

1. **`VideoSource`** — unified interface over webcam, file, or RTSP URL with auto-reconnect
2. **`Detector`** — thin YOLO wrapper; filters to COCO vehicle classes (car, motorcycle, bus, truck)
3. **`CentroidTracker`** — Kalman-filter nearest-neighbour tracker; tracks that exceed `max_missing` frames move to a *graveyard* where they can be resurrected by ReID
4. **`SpeedEstimator`** — converts frame-to-frame centroid displacement to speed using a depth-aware scale map derived from the calibration points; rolling window + spike rejection
5. **`Overlay`** — draws bounding boxes, speed labels, and the calibration track onto frames

### Calibration

Calibration maps pixel coordinates to metres. The user places N points along the road with known real-world distances between adjacent segments. Each segment gives a `metres_per_pixel` value at its midpoint y-coordinate. At runtime, `SpeedEstimator._mpp_at(y)` interpolates between segments so scale varies correctly with depth.

Calibrations are stored in `calibration.json` (in `~/.speedcam` for packaged builds, or the working directory when running from source) keyed by a stable source identifier:

```json
{
  "webcam:0": {
    "points": [{"x": 960, "y": 540}, {"x": 1200, "y": 700}],
    "distances": [12.0],
    "frame_w": 1920,
    "frame_h": 1080,
    "created_at": "2025-01-01T00:00:00+00:00"
  }
}
```

### Data directory

`speedcam.core.get_data_dir()` returns the writable data directory. It reads `SPEEDCAM_DATA_DIR` from the environment (set by `launcher.py` for packaged builds), falling back to the current working directory. All calibration reads/writes and upload temp files use this path, so the app is safe to run from a read-only location.

---

## Tests

```bash
pip install pytest pytest-flask
pytest tests/ -v
```

Tests run without a GPU, camera, or model files — `ultralytics` is stubbed in `conftest.py`. The `isolated_data_dir` fixture points `SPEEDCAM_DATA_DIR` at a fresh `tmp_path` for each test so calibration writes never touch your working directory.

For CI, `requirements-ci.txt` omits `ultralytics` and `torch` entirely, keeping the runner install under 200 MB.

---

## Building releases

### Desktop apps

```bash
pip install pyinstaller
# Download the model first (PyInstaller bundles whatever is in the project root)
python -c "from ultralytics import YOLO; YOLO('yolo12s.pt')"
pyinstaller speedometer.spec --noconfirm
```

Outputs:
- **Mac**: `dist/Speedometer.app` → wrap with `hdiutil create`
- **Windows**: `dist/Speedometer/` → wrap with `makensis build/installer.nsi`
- **Linux**: `dist/Speedometer/` → `tar -czf`

### Docker

```bash
docker build -t speedometer .
docker run -p 8080:8080 -v speedometer-data:/root/.speedcam speedometer
```

### Releasing via GitHub Actions

Push a version tag to trigger all four platform builds automatically:

```bash
git tag v1.0.0
git push origin v1.0.0
```

The `release.yml` workflow builds Mac DMG, Windows installer, Linux tarball, and Docker image, then attaches everything to the GitHub Release.

---

## API reference

All endpoints are defined in `speedcam/web/routes.py`.

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Web UI |
| `GET` | `/video_feed` | MJPEG stream |
| `GET` | `/preview_frame` | Single JPEG for calibration |
| `POST` | `/api/upload` | Upload a video file |
| `POST` | `/api/grab_webcam_frame` | Capture a webcam frame for preview |
| `POST` | `/api/calibrate` | Save calibration points |
| `POST` | `/api/load_calibration` | Load saved webcam calibration |
| `POST` | `/api/start` | Start the detection pipeline |
| `POST` | `/api/stop` | Stop the pipeline |
| `POST` | `/api/reset` | Stop + clear all stats |
| `GET` | `/api/stats` | Current speed stats and detection list |
| `GET/POST` | `/api/map_segment` | Get/set GPS road segment |
| `GET` | `/api/download_csv` | Export detections as CSV |
