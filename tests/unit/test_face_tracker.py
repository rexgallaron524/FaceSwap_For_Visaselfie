from types import SimpleNamespace

import numpy as np
import pytest

from app.pipeline.types import TrackingStatus, VideoFrame
from app.tracking.mediapipe_tracker import MediaPipeFaceTracker


class FakeBackend:
    def __init__(self, _model_path, callback):
        self.callback = callback
        self.submissions = []
        self.closed = False

    def submit(self, rgb, timestamp_ms):
        self.submissions.append((rgb, timestamp_ms))

    def complete(self, result):
        self.callback(result, None, self.submissions[-1][1])

    def close(self):
        self.closed = True


def video_frame(frame_id: int, timestamp_ns: int) -> VideoFrame:
    rgb = np.zeros((720, 1280, 3), dtype=np.uint8)
    rgb.setflags(write=False)
    return VideoFrame(frame_id, timestamp_ns, rgb)


def face_result():
    landmarks = []
    for index in range(478):
        angle = 2 * np.pi * index / 478
        landmarks.append(
            SimpleNamespace(
                x=0.5 + 0.16 * np.cos(angle),
                y=0.5 + 0.22 * np.sin(angle),
                z=0.01 * np.sin(angle),
            )
        )
    categories = [
        SimpleNamespace(category_name="mouthSmileLeft", display_name="", score=0.7),
        SimpleNamespace(category_name="eyeBlinkRight", display_name="", score=0.2),
        SimpleNamespace(category_name="jawOpen", display_name="", score=0.4),
    ]
    return SimpleNamespace(
        face_landmarks=[landmarks],
        facial_transformation_matrixes=[np.eye(4)],
        face_blendshapes=[categories],
    )


def test_tracker_keeps_one_frame_in_flight_and_converts_backend_result(tmp_path):
    model = tmp_path / "model.task"
    model.write_bytes(b"model")
    clock = iter((1_000_000_000, 1_025_000_000))
    created = []

    def factory(path, callback):
        backend = FakeBackend(path, callback)
        created.append(backend)
        return backend

    tracker = MediaPipeFaceTracker(model, monotonic_ns=lambda: next(clock), backend_factory=factory)
    tracker.open()

    assert tracker.submit(video_frame(10, 2_000_000_000))
    assert not tracker.submit(video_frame(11, 2_033_000_000))
    created[0].complete(face_result())
    result = tracker.poll_latest()

    assert result.status is TrackingStatus.TRACKED
    assert result.frame_id == 10
    assert result.latency_ms == pytest.approx(25.0)
    assert result.face.landmark_schema == "mediapipe-face-landmarker-478-v1"
    assert len(result.face.landmarks) == len(result.face.normalized_landmarks) == 478
    assert result.face.bounds.width == pytest.approx(0.32 * 1280, rel=1e-3)
    assert result.face.scale == pytest.approx(0.32, rel=1e-3)
    assert result.face.pose.yaw == pytest.approx(0.0)
    assert result.face.tracking_confidence == pytest.approx(1.0)
    assert result.face.blendshapes["mouth_smile_left"] == pytest.approx(0.7)
    assert tracker.poll_latest() is None
    tracker.close()
    assert created[0].closed


def test_tracker_reports_no_face_as_a_completed_result(tmp_path):
    model = tmp_path / "model.task"
    model.write_bytes(b"model")
    created = []

    def factory(path, callback):
        backend = FakeBackend(path, callback)
        created.append(backend)
        return backend

    times = iter((5_000_000_000, 5_010_000_000))
    tracker = MediaPipeFaceTracker(model, monotonic_ns=lambda: next(times), backend_factory=factory)
    tracker.open()
    assert tracker.submit(video_frame(0, 1_000_000_000))
    created[0].complete(
        SimpleNamespace(face_landmarks=[], facial_transformation_matrixes=[], face_blendshapes=[])
    )

    result = tracker.poll_latest()
    assert result.status is TrackingStatus.NO_FACE
    assert result.face is None
    assert result.latency_ms == pytest.approx(10.0)
