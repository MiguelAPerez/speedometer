"""Unit tests for SpeedEstimator — no video, no GPU required."""
from __future__ import annotations

import pytest
from speedcam.speed import SpeedEstimator
from speedcam.tracker import Track


def _track(tid: int, cx: float, cy: float, *, detected: bool = True) -> Track:
    """Minimal Track instance for feeding into SpeedEstimator."""
    return Track(
        track_id=tid,
        centroid=(cx, cy),
        predicted=(cx, cy),
        bbox=(cx - 10, cy - 10, cx + 10, cy + 10),
        cls=2,
        conf=0.9,
        missing_frames=0,
        detected_this_frame=detected,
    )


def _feed(estimator: SpeedEstimator, positions: list[tuple[float, float]],
          fps: float = 10.0, tid: int = 0) -> dict:
    """Feed a sequence of (cx, cy) positions and return the last result dict."""
    result = {}
    for i, (cx, cy) in enumerate(positions):
        result = estimator.update({tid: _track(tid, cx, cy)}, frame_ts=i / fps)
    return result


class TestSpeedEstimator:
    def test_rightward_speed(self):
        # 10 px/frame at 10 fps, 0.1 m/px → 10 m/s → ~22.4 mph
        est = SpeedEstimator(scale_pts=[(240, 0.1)], min_samples=3)
        positions = [(100.0 + i * 10, 240.0) for i in range(6)]
        result = _feed(est, positions)

        assert 0 in result
        rec = result[0]
        assert rec.direction == "→"
        assert 20.0 < rec.speed_mph < 25.0

    def test_leftward_direction(self):
        est = SpeedEstimator(scale_pts=[(240, 0.1)], min_samples=3)
        positions = [(300.0 - i * 10, 240.0) for i in range(6)]
        result = _feed(est, positions)

        assert 0 in result
        assert result[0].direction == "←"

    def test_stationary_reads_zero(self):
        # Jitter below min_speed_mph should clamp to 0
        est = SpeedEstimator(scale_pts=[(240, 0.001)], min_samples=3, min_speed_mph=2.0)
        positions = [(100.0 + (i % 2) * 0.5, 240.0) for i in range(8)]
        result = _feed(est, positions)

        if 0 in result:
            assert result[0].speed_mph == 0.0

    def test_no_result_below_min_samples(self):
        est = SpeedEstimator(scale_pts=[(240, 0.1)], min_samples=5)
        # Only 3 frames — should not yet emit a result
        positions = [(100.0 + i * 10, 240.0) for i in range(3)]
        result = _feed(est, positions)
        assert 0 not in result

    def test_spike_rejection(self):
        # Build a ~5 px/frame baseline then inject a 1000 px jump.
        # The spike should be rejected; speed should stay near baseline.
        est = SpeedEstimator(scale_pts=[(240, 0.05)], min_samples=3, min_speed_mph=0.1)
        for i in range(8):
            est.update({0: _track(0, 100.0 + i * 5, 240.0)}, frame_ts=i * 0.1)

        spike = est.update({0: _track(0, 100.0 + 8 * 5 + 1000, 240.0)}, frame_ts=0.9)
        if 0 in spike:
            # 1000 px * 0.05 m/px / 0.1 s = 500 m/s → should be rejected
            assert spike[0].speed_mps < 50.0

    def test_reset_clears_state(self):
        est = SpeedEstimator(scale_pts=[(240, 0.1)], min_samples=3)
        positions = [(100.0 + i * 10, 240.0) for i in range(6)]
        _feed(est, positions)
        est.reset()
        # After reset a single frame should produce no output
        result = est.update({0: _track(0, 200.0, 240.0)}, frame_ts=0.0)
        assert 0 not in result

    def test_from_track_factory(self):
        points = [{"x": 0, "y": 200}, {"x": 100, "y": 200}]
        distances = [10.0]  # 10 m over 100 px → 0.1 m/px
        est = SpeedEstimator.from_track(points, distances)
        assert len(est._scale_pts) == 1
        assert abs(est._scale_pts[0][1] - 0.1) < 1e-9

    def test_from_track_requires_two_points(self):
        with pytest.raises(ValueError):
            SpeedEstimator.from_track([{"x": 0, "y": 0}], [])

    def test_from_track_zero_length_segment_raises(self):
        points = [{"x": 5, "y": 5}, {"x": 5, "y": 5}]
        with pytest.raises(ValueError):
            SpeedEstimator.from_track(points, [10.0])

    def test_multi_segment_interpolation(self):
        # Two segments at different depths — mpp should vary by y-position
        points = [{"x": 0, "y": 100}, {"x": 100, "y": 100}, {"x": 200, "y": 300}]
        distances = [5.0, 10.0]
        est = SpeedEstimator.from_track(points, distances)
        assert len(est._scale_pts) == 2
        # mpp at top of frame should match segment 0
        mpp_top = est._mpp_at(0)
        mpp_bot = est._mpp_at(400)
        assert mpp_top == est._scale_pts[0][1]
        assert mpp_bot == est._scale_pts[-1][1]

    def test_kalman_only_frames_skipped(self):
        # Frames with detected_this_frame=False should not advance the speed window
        est = SpeedEstimator(scale_pts=[(240, 0.1)], min_samples=3)
        est.update({0: _track(0, 100.0, 240.0, detected=False)}, frame_ts=0.0)
        result = est.update({0: _track(0, 200.0, 240.0, detected=False)}, frame_ts=0.1)
        assert 0 not in result
