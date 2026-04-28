"""Unit tests for calibration I/O and source helpers in speedcam.core."""
from __future__ import annotations

import json

import pytest
from speedcam.core import (
    clear_calibration,
    get_data_dir,
    is_live_camera,
    load_calibration,
    save_calibration,
    source_key,
)

# ---------------------------------------------------------------------------
# source_key
# ---------------------------------------------------------------------------

def test_source_key_int():
    assert source_key(0) == "webcam:0"
    assert source_key(1) == "webcam:1"


def test_source_key_digit_string():
    assert source_key("0") == "webcam:0"
    assert source_key("2") == "webcam:2"


def test_source_key_url():
    url = "rtsp://192.168.1.10:8554/cam"
    assert source_key(url) == url


# ---------------------------------------------------------------------------
# is_live_camera
# ---------------------------------------------------------------------------

def test_is_live_camera_int():
    assert is_live_camera(0) is True


def test_is_live_camera_file(tmp_path):
    f = tmp_path / "clip.mp4"
    f.touch()
    assert is_live_camera(str(f)) is False


def test_is_live_camera_url():
    assert is_live_camera("rtsp://cam/stream") is False


# ---------------------------------------------------------------------------
# get_data_dir
# ---------------------------------------------------------------------------

def test_get_data_dir_env(tmp_path, monkeypatch):
    monkeypatch.setenv("SPEEDCAM_DATA_DIR", str(tmp_path))
    assert get_data_dir() == tmp_path


def test_get_data_dir_default(monkeypatch):
    monkeypatch.delenv("SPEEDCAM_DATA_DIR", raising=False)
    from pathlib import Path
    assert get_data_dir() == Path(".")


# ---------------------------------------------------------------------------
# save / load / clear calibration  (isolated_data_dir fixture keeps each test
# writing to its own tmp dir via SPEEDCAM_DATA_DIR)
# ---------------------------------------------------------------------------

_PTS  = [{"x": 10.0, "y": 100.0}, {"x": 200.0, "y": 100.0}]
_DIST = [15.0]


def test_save_and_load_roundtrip():
    save_calibration(0, _PTS, _DIST, 1920, 1080)
    cal = load_calibration(0, 1920, 1080)

    assert cal is not None
    assert cal["distances"] == _DIST
    assert cal["frame_w"] == 1920
    assert cal["frame_h"] == 1080
    assert len(cal["points"]) == 2
    assert cal["points"][0]["x"] == pytest.approx(10.0)


def test_load_calibration_scales_to_new_resolution():
    save_calibration(0, _PTS, _DIST, 1920, 1080)
    cal = load_calibration(0, 960, 540)  # half size

    assert cal["points"][0]["x"] == pytest.approx(5.0)
    assert cal["points"][0]["y"] == pytest.approx(50.0)
    assert cal["points"][1]["x"] == pytest.approx(100.0)


def test_clear_calibration_removes_entry():
    save_calibration(0, _PTS, _DIST, 640, 480)
    clear_calibration(0)
    assert load_calibration(0, 640, 480) is None


def test_load_nonexistent_source_returns_none():
    assert load_calibration(99, 640, 480) is None


def test_multiple_sources_stored_independently():
    pts2 = [{"x": 50.0, "y": 200.0}, {"x": 150.0, "y": 200.0}]
    save_calibration(0, _PTS, [10.0], 1280, 720)
    save_calibration(1, pts2, [20.0], 1280, 720)

    cal0 = load_calibration(0, 1280, 720)
    cal1 = load_calibration(1, 1280, 720)

    assert cal0["distances"] == [10.0]
    assert cal1["distances"] == [20.0]
    assert cal1["points"][0]["x"] == pytest.approx(50.0)


def test_calibration_json_written_to_data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("SPEEDCAM_DATA_DIR", str(tmp_path))
    save_calibration(0, _PTS, _DIST, 640, 480)
    cal_file = tmp_path / "calibration.json"
    assert cal_file.exists()
    data = json.loads(cal_file.read_text())
    assert "webcam:0" in data
