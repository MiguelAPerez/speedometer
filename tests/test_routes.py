"""Integration tests for Flask routes — no GPU, no real video required."""
from __future__ import annotations

import io
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Basic page / API smoke tests
# ---------------------------------------------------------------------------

def test_index_returns_200(client):
    r = client.get("/")
    assert r.status_code == 200


def test_stats_default_state(client):
    r = client.get("/api/stats")
    assert r.status_code == 200
    data = r.get_json()
    assert data["running"] is False
    assert data["last_speed"] is None
    assert data["avg_speed"] is None
    assert data["count"] == 0
    assert "units" in data


def test_stop_always_succeeds(client):
    r = client.post("/api/stop")
    assert r.status_code == 200
    assert r.get_json()["ok"] is True


def test_reset_clears_stats(client):
    r = client.post("/api/reset")
    assert r.status_code == 200
    assert r.get_json()["ok"] is True


def test_download_csv_empty(client):
    r = client.get("/api/download_csv")
    assert r.status_code == 404


def test_start_without_calibration_returns_400(client):
    r = client.post("/api/start", json={"units": "mph"})
    assert r.status_code == 400
    assert "calibration" in r.get_json()["error"].lower()


def test_start_already_running_returns_400(client):
    from speedcam.web import state
    with state._lock:
        state._state["running"] = True
        state._state["calibration"] = {"points": [], "distances": []}

    r = client.post("/api/start", json={"units": "mph"})
    assert r.status_code == 400


# ---------------------------------------------------------------------------
# Calibration endpoint
# ---------------------------------------------------------------------------

def test_calibrate_valid(client):
    payload = {
        "points": [{"x": 0, "y": 100}, {"x": 200, "y": 100}],
        "distances": [15.0],
        "frame_w": 1920,
        "frame_h": 1080,
    }
    r = client.post("/api/calibrate", json=payload)
    assert r.status_code == 200
    assert r.get_json()["ok"] is True


def test_calibrate_too_few_points(client):
    payload = {
        "points": [{"x": 0, "y": 0}],
        "distances": [],
        "frame_w": 640,
        "frame_h": 480,
    }
    r = client.post("/api/calibrate", json=payload)
    assert r.status_code == 400


def test_calibrate_mismatched_distances(client):
    payload = {
        "points": [{"x": 0, "y": 0}, {"x": 100, "y": 0}],
        "distances": [5.0, 10.0],  # one too many
        "frame_w": 640,
        "frame_h": 480,
    }
    r = client.post("/api/calibrate", json=payload)
    assert r.status_code == 400


# ---------------------------------------------------------------------------
# Upload endpoint (VideoSource mocked)
# ---------------------------------------------------------------------------

def test_upload_no_file_returns_400(client):
    r = client.post("/api/upload")
    assert r.status_code == 400


def test_upload_success(client):
    fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_vs = MagicMock()
    mock_vs.grab_single_frame.return_value = fake_frame

    with patch("speedcam.web.routes.VideoSource", return_value=mock_vs):
        data = {"video": (io.BytesIO(b"fake"), "test.mp4")}
        r = client.post("/api/upload", data=data, content_type="multipart/form-data")

    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True
    assert body["w"] == 640
    assert body["h"] == 480


def test_upload_unreadable_video(client):
    mock_vs = MagicMock()
    mock_vs.grab_single_frame.return_value = None

    with patch("speedcam.web.routes.VideoSource", return_value=mock_vs):
        data = {"video": (io.BytesIO(b"bad"), "bad.mp4")}
        r = client.post("/api/upload", data=data, content_type="multipart/form-data")

    assert r.status_code == 500


# ---------------------------------------------------------------------------
# Map segment endpoint
# ---------------------------------------------------------------------------

def test_set_map_segment(client):
    payload = {
        "points": [{"lat": 34.05, "lng": -118.25}, {"lat": 34.06, "lng": -118.24}],
        "label": "Main St",
    }
    r = client.post("/api/map_segment", json=payload)
    assert r.status_code == 200
    assert r.get_json()["ok"] is True


def test_get_map_segment_empty(client):
    r = client.get("/api/map_segment")
    assert r.status_code == 200
    data = r.get_json()
    assert "segment" in data
