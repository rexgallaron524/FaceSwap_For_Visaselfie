import numpy as np
import pytest

from app.camera.metrics import CaptureMetrics
from app.pipeline.types import VideoFrame


def frame(frame_id: int, timestamp_ns: int) -> VideoFrame:
    pixels = np.zeros((1, 1, 3), dtype=np.uint8)
    pixels.setflags(write=False)
    return VideoFrame(frame_id, timestamp_ns, pixels)


def test_capture_fps_uses_source_ids_to_account_for_skipped_preview_frames():
    metrics = CaptureMetrics(window_seconds=2)
    metrics.observe(frame(10, 1_000_000_000))
    metrics.observe(frame(12, 1_100_000_000))
    metrics.observe(frame(13, 1_200_000_000))

    assert metrics.fps == pytest.approx(15.0)
    assert metrics.preview_drops == 1


def test_new_capture_session_resets_on_non_increasing_frame_id():
    metrics = CaptureMetrics()
    metrics.observe(frame(5, 1_000_000_000))
    metrics.observe(frame(0, 2_000_000_000))

    assert metrics.fps is None
    assert metrics.preview_drops == 0


def test_invalid_metrics_window_is_rejected():
    with pytest.raises(ValueError):
        CaptureMetrics(0)
