from __future__ import annotations

import sys
from unittest.mock import MagicMock

# Stub out ultralytics before any speedcam module is imported.
# detector.py defers the import to Detector.__init__, but stubbing here
# prevents accidental real imports if something imports at module level.
sys.modules.setdefault("ultralytics", MagicMock())

import pytest
from app import app as _flask_app


@pytest.fixture()
def app():
    _flask_app.config["TESTING"] = True
    yield _flask_app


@pytest.fixture()
def client(app):
    return app.test_client()


@pytest.fixture(autouse=True)
def isolated_data_dir(tmp_path, monkeypatch):
    """Point SPEEDCAM_DATA_DIR at a fresh tmp dir for every test."""
    monkeypatch.setenv("SPEEDCAM_DATA_DIR", str(tmp_path))
    (tmp_path / "tmp").mkdir()


@pytest.fixture(autouse=True)
def reset_app_state():
    """Reset the in-memory pipeline state between tests."""
    from speedcam.web import state

    yield

    state.reset_stats()
    with state._lock:
        state._state["running"] = False
        state._state["tmp_video_path"] = None
        state._state["calibration"] = None
        state._state["preview_frame"] = None
        state._state["pipeline_thread"] = None
        state._state["error"] = None
        state._state["source"] = None
