## v1.2.0 (2026-04-29)

### Feat

- **perf**: add hardware acceleration and device optimization

### Fix

- **desktop**: PyInstaller 6.x path handling and native window integration
- patching nvr link
- **pipeline**: use blocking put for video files to prevent frame drop race

### Refactor

- **save+start**: move to 2 different buttons

## v1.1.0 (2026-04-28)

### Feat

- productionize for desktop and Docker distribution
- add peak-speed tracking with thumbnails and offline Leaflet
- UI polish + Leaflet street map tab with detection overlay (#1)
- add pipeline factory to centralize detector/tracker defaults
- add car re-identification system with graveyard memory and upgrade to yolo11s
- enhance calibration system with x-coordinates; update speed estimation and drawing functions
- add speed detection system with calibration and tracking

### Fix

- prevent 0 mph from freezing KPI panel values; use rolling speed buffer
- prevent spike rejection from blocking car acceleration from rest
- clear canvas calibration overlay and fill speed estimation warmup gap
- use shared pipeline factory in test_tracking.py instead of hardcoded defaults
- reduce false positives and double-detections in vehicle detection
- update default port to 8080 and enhance calibration markers; add speed clamping for low speeds

### Refactor

- implement adaptive polling for stats with smart scheduling
- break monolithic app.py into modular structure
