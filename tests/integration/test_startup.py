"""Exercise the actual entry point and Qt event loop without camera hardware."""

import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("module", ["app", "app.main"])
def test_application_starts_and_exits_cleanly(tmp_path, module):
    # Run outside the repo to verify the installed package/defaults are CWD independent.
    result = subprocess.run(
        [sys.executable, "-m", module, "--smoke-test"],
        cwd=tmp_path,
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen", "LOCALAPPDATA": str(tmp_path)},
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    log = (tmp_path / "FaceLive" / "logs" / "facelive.log").read_text(encoding="utf-8")
    assert "Application shell ready; geometric face replacement preview available" in log
    assert "Application stopped (exit 0)" in log


def test_invalid_config_fails_before_ui_startup(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "app", "--config", str(tmp_path / "missing.toml")],
        cwd=tmp_path,
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen", "LOCALAPPDATA": str(tmp_path)},
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 2
    assert "Cannot read configuration" in result.stderr
    assert not (tmp_path / "FaceLive" / "logs").exists()


def test_shell_starts_and_stops_deterministic_camera_preview(tmp_path):
    script = """
import time
import numpy as np
from PySide6.QtWidgets import QLabel, QProgressBar
from app.config import AppConfig
from app.main import create_application
from app.pipeline.types import (
    CameraDevice, FaceState, FrameFormat, HeadPose, NormalizedLandmark, Point2D,
    Rect, StageError, TrackingResult, TrackingStatus, VideoFrame,
)
from app.ui.reference_dialog import ReferenceEnrollmentDialog
class FakeSource:
    def __init__(self):
        self.frames = 0
        self.closed = False
        self.fail = False
        self.last_pixels = None
    def enumerate_devices(self):
        return (CameraDevice('camera-1', 'Test Camera', False),)
    def open(self, device_id, requested):
        assert device_id == 'camera-1'
        return FrameFormat(2, 1, 30)
    def read_latest(self):
        if self.fail:
            raise StageError('Test camera disconnected')
        if self.frames:
            return None
        self.frames += 1
        pixels = np.array([[[255, 0, 0], [0, 255, 0]]], dtype=np.uint8)
        pixels.setflags(write=False)
        self.last_pixels = pixels
        return VideoFrame(0, 1_000_000_000, pixels)
    def close(self):
        self.closed = True
class FakeTracker:
    def __init__(self):
        self.latest = None
        self.closed = False
    def open(self):
        self.closed = False
    def submit(self, frame):
        face = FaceState(
            frame.frame_id, frame.timestamp_ns, Rect(0.0, 0.0, 2.0, 1.0),
            Point2D(1.0, 0.5), 1.0, HeadPose(12.5, -3.0, 1.5),
            (Point2D(1.0, 0.5),), (NormalizedLandmark(0.5, 0.5, 0.0),),
            'test-landmarks-v1', 0.9,
            {'mouth_smile_left': 0.6, 'mouth_smile_right': 0.8,
             'eye_blink_left': 0.1, 'eye_blink_right': 0.2, 'jaw_open': 0.3},
        )
        self.latest = TrackingResult(
            frame.frame_id, frame.timestamp_ns, TrackingStatus.TRACKED,
            face=face, latency_ms=4.2,
        )
        return True
    def poll_latest(self):
        result, self.latest = self.latest, None
        return result
    def close(self):
        self.closed = True
source = FakeSource()
tracker = FakeTracker()
application, window = create_application(AppConfig(), lambda: source, lambda: tracker)
window.show()
application.processEvents()
assert window.windowTitle() == 'FaceLive'
assert window.isVisible()
assert window._layout_mode == 'wide'
assert window.camera_card.geometry().right() < window.preview_card.geometry().left()
assert window.preview_card.geometry().right() < window.diagnostics_card.geometry().left()
metric_labels = [
    label for label in window.diagnostics_card.findChildren(QLabel)
    if label.objectName() == 'metricLabel'
]
assert len(metric_labels) == 15
assert all(label.height() >= label.fontMetrics().height() for label in metric_labels)
assert all(
    bar.minimumHeight() >= bar.fontMetrics().height()
    for bar in window.reference_weight_view.findChildren(QProgressBar)
)
window.resize(1380, 900)
application.processEvents()
assert window._layout_mode == 'wide'
window.resize(1330, 900)
application.processEvents()
assert window._layout_mode == 'medium'
window.resize(1380, 900)
application.processEvents()
assert window._layout_mode == 'medium'
window.resize(1100, 800)
application.processEvents()
assert window._layout_mode == 'medium'
assert window.camera_card.geometry().right() < window.preview_card.geometry().left()
assert window.preview_card.geometry().bottom() < window.diagnostics_card.geometry().top()
window.resize(760, 700)
application.processEvents()
assert window._layout_mode == 'compact'
assert window.camera_card.geometry().bottom() < window.preview_card.geometry().top()
assert window.preview_card.geometry().bottom() < window.reference_library_card.geometry().top()
assert window._content_scroll.verticalScrollBar().maximum() > 0
window.resize(1440, 900)
application.processEvents()
assert window._layout_mode == 'wide'
assert window.camera_selector.isEnabled()
assert window.camera_selector.currentText() == 'Test Camera'
assert window.capture_button.isEnabled()
window.start_capture()
deadline = time.monotonic() + 2
while not window._capturing and time.monotonic() < deadline:
    application.processEvents()
    time.sleep(0.01)
assert window._capturing
window._poll_camera()
application.processEvents()
assert window.capture_state_label.text() == 'Running'
assert window.tracking_state_label.text() == 'Tracked'
assert window.tracking_confidence_label.text() == '90% derived'
assert window.head_pose_label.text() == '+12.5° · -3.0° · +1.5°'
assert window.smile_label.text() == '70%'
assert window.eye_closure_label.text() == '10% · 20%'
assert window.mouth_activity_label.text() == '30% open'
assert window.tracking_latency_label.text() == '4.2 ms'
assert window.reference_selection_status.text() == 'No enrolled references'
assert len(window.reference_weight_view.displayed_weights()) == 8
assert set(window.reference_weight_view.displayed_weights().values()) == {0.0}
assert window.preview_image.pixmap() is not None
assert window.preview_mode_selector.currentData() == 'diagnostic'
assert window.preview_mode_selector.count() == 3
assert window.mirror_preview_toggle.isChecked()
assert window._last_image.pixelColor(0, 0).red() == 255
mirrored = window._oriented_preview_image()
assert mirrored.pixelColor(0, 0).green() == 255
window.mirror_preview_toggle.setChecked(False)
unmirrored = window._oriented_preview_image()
assert unmirrored.pixelColor(0, 0).red() == 255
assert source.last_pixels.tolist() == [[[255, 0, 0], [0, 255, 0]]]
source.fail = True
window._poll_camera()
assert window.capture_state_label.text() == 'Disconnected'
assert window.capture_button.isEnabled()
source.fail = False
source.frames = 0
window.refresh_cameras()
window.start_capture()
deadline = time.monotonic() + 2
while not window._capturing and time.monotonic() < deadline:
    application.processEvents()
    time.sleep(0.01)
assert window._capturing
assert window.load_references.isEnabled()
assert window.reference_summary_label.text() == 'Incomplete · 0 of 8 required'
dialog = ReferenceEnrollmentDialog(window._reference_session, window)
dialog.show()
application.processEvents()
assert len(dialog._cards) == 8
assert set(dialog._cards) == {
    'front-neutral', 'front-smile', 'left-20', 'left-40',
    'right-20', 'right-40', 'up', 'down'
}
assert dialog.summary.text() == '0 of 8 required references valid'
assert not dialog.save_button.isEnabled()
assert dialog.save_button.text() == 'Save as…'
assert dialog.location.text().startswith('Will save automatically to ')
dialog.close()
assert not window.replacement_toggle.isEnabled()
assert not window.replacement_toggle.isChecked()
assert not window.virtual_camera_button.isEnabled()
window.close()
application.processEvents()
assert not window.isVisible()
assert source.closed
assert tracker.closed
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(tmp_path),
        env={
            **os.environ,
            "QT_QPA_PLATFORM": "offscreen",
            "LOCALAPPDATA": str(tmp_path),
        },
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
