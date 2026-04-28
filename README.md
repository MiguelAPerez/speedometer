# Speedometer

Measure the speed of passing vehicles using any camera — no special hardware required.  
Point a camera at a road, mark two reference points, and the app tracks vehicles in real time from your browser.

---

## Download

Grab the latest release for your platform:

| Platform | Download |
|----------|----------|
| **Mac** | `Speedometer-vX.X.X-mac.dmg` |
| **Windows** | `Speedometer-vX.X.X-windows-setup.exe` |
| **Linux** | `Speedometer-vX.X.X-linux.tar.gz` |
| **Self-host (Docker)** | See [Docker](#docker--self-hosting) below |

[**→ All releases**](../../releases/latest)

---

## Installation

### Mac
1. Open the `.dmg` file
2. Drag **Speedometer** into your Applications folder
3. Double-click to launch — your browser opens automatically

> First launch may take 30–60 seconds while the AI model loads.

### Windows
1. Run the `.exe` installer and follow the prompts
2. Launch **Speedometer** from the desktop shortcut or Start Menu
3. Your browser opens automatically to the app

### Linux
1. Extract the `.tar.gz` archive
2. Run `./run.sh` inside the extracted folder
3. Your browser opens automatically to the app

---

## How to use it

### 1 — Choose a video source

- **Upload a video** — drag in any `.mp4`, `.mov`, or `.avi` file
- **Use your webcam** — click **Use Webcam** to grab a live frame

### 2 — Calibrate the scene

The app needs to know how large real-world distances are in your footage.

1. Click **two points** on the road that are a known distance apart  
   *(painted lines, lamp-post spacing, or a tape measure work well — aim for 8–15 m)*
2. Enter the real-world distance between the points in metres
3. Click **Save Calibration**

Calibration is saved automatically — you only need to redo it if you move the camera.

### 3 — Start detecting

Click **Start**. Vehicles are highlighted with bounding boxes and their speeds appear live.  
Switch between **mph** and **km/h** at any time.

### 4 — Export results

Click **Download CSV** to export a spreadsheet of all detections with timestamps, vehicle type, speed, and direction.

---

## Camera placement tips

- **Height**: Mount 2–5 m above the road, angled slightly downward
- **Angle**: Vehicles should travel roughly perpendicular to the camera for best accuracy
- **Lighting**: Avoid heavy backlighting or overexposed frames
- **Typical accuracy**: ±2–3 mph at 30–60 mph in good conditions

---

## Docker / Self-hosting

Run Speedometer on a server or NAS and access it from any browser on your network.

**Quick start:**
```bash
docker compose up
```
Then open `http://your-server-ip:8080`.

**Pull the image manually:**
```bash
docker run -p 8080:8080 -v speedometer-data:/root/.speedcam \
  ghcr.io/miguelaperez/speedometer:latest
```

Data (calibration, uploaded videos) is stored in the `speedometer-data` volume and persists across restarts.

---

## Where data is stored

All data lives in `~/.speedcam` on your machine (or the Docker volume):

| File | Contents |
|------|----------|
| `calibration.json` | Your saved calibration points |
| `speedometer.log` | Error log (only written on problems) |
| `tmp/` | Temporary upload files (auto-cleaned) |

Nothing is sent to the internet. No telemetry, no accounts.

---

## Troubleshooting

**The app opens but the video feed is blank**  
Check that your browser has camera permission if using the webcam. For uploaded videos, make sure the file isn't open in another app.

**"Model loading" takes a long time on first launch**  
The AI model (~18 MB) is set up on first run. Subsequent launches are fast.

**Speeds look wrong**  
Recalibrate — even a small change in camera position requires a new calibration. Ensure your two reference points are as far apart as possible for best accuracy.

**Port 8080 is already in use**  
The app automatically tries nearby ports. Check the launcher window for the actual URL.

---

## For developers

See [docs/development.md](docs/development.md) for architecture, running from source, contributing, and CLI usage.
