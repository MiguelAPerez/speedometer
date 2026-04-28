"""
app.py — Flask entry point for speedcam.

Run with:  python app.py
Then open:  http://localhost:8080
"""

from __future__ import annotations

import argparse
from pathlib import Path

from flask import Flask

from speedcam.web.routes import bp

app = Flask(__name__, template_folder="templates")
app.register_blueprint(bp)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="speedcam web UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    Path("tmp").mkdir(exist_ok=True)

    print(f"speedcam running at http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)
