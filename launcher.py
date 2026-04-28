"""
launcher.py — Desktop entry point for Speedometer.

Double-click (or run) to start the app. Opens the web UI in your default
browser automatically. Data is stored in ~/.speedcam so it survives updates.
"""
from __future__ import annotations

import logging
import os
import shutil
import socket
import sys
import threading
import time
import webbrowser
from pathlib import Path


def _base_dir() -> Path:
    """Directory containing this executable (or this script during dev)."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).parent


def _find_free_port(start: int = 8080) -> int:
    for port in range(start, start + 20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    return start


def _setup_data_dir(base: Path) -> Path:
    data = Path.home() / ".speedcam"
    data.mkdir(exist_ok=True)
    (data / "tmp").mkdir(exist_ok=True)
    return data


def _copy_models(base: Path, data: Path) -> None:
    """Copy bundled model files to the data dir on first launch."""
    for name in ("yolov8n.pt", "yolo11s.pt", "yolo12s.pt"):
        src = base / name
        dst = data / name
        if src.exists() and not dst.exists():
            print(f"Installing model {name}…")
            shutil.copy2(src, dst)


def _configure_logging(data: Path) -> None:
    log_file = data / "speedometer.log"
    logging.basicConfig(
        filename=str(log_file),
        level=logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # Silence noisy werkzeug request logs
    logging.getLogger("werkzeug").setLevel(logging.ERROR)


def main() -> None:
    base = _base_dir()
    data = _setup_data_dir(base)
    _copy_models(base, data)
    _configure_logging(data)

    os.environ["SPEEDCAM_DATA_DIR"] = str(data)
    os.chdir(data)

    # Ensure the speedcam package is importable when running frozen
    if str(base) not in sys.path:
        sys.path.insert(0, str(base))

    port = _find_free_port(8080)
    url = f"http://127.0.0.1:{port}"

    from flask import Flask
    from speedcam.web.routes import bp

    flask_app = Flask(
        __name__,
        template_folder=str(base / "templates"),
        static_folder=str(base / "static"),
    )
    flask_app.register_blueprint(bp)

    server_thread = threading.Thread(
        target=lambda: flask_app.run(
            host="127.0.0.1", port=port, debug=False, threaded=True
        ),
        daemon=True,
    )
    server_thread.start()

    # Give Flask a moment to bind before the browser hits it
    time.sleep(1.2)
    webbrowser.open(url)

    print(f"Speedometer running at {url}")
    print("Close this window to stop.")

    try:
        server_thread.join()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
