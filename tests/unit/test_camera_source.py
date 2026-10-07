import itertools
import time
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from app.camera.opencv_source import OpenCVCameraSource
from app.pipeline.types import FrameFormat, StageError


class FakeCapture:
    def __init__(self, _index, _backend, *, fail_after_first=False):
        self.opened = True
        self.released = False
        self.fail_after_first = fail_after_first
        self.read_count = 0
        self.properties = {}

    def isOpened(self):  # noqa: N802
        return self.opened

    def set(self, property_id, value):
        self.properties[property_id] = value
        return True

    def get(self, property_id):
        return {
            cv2.CAP_PROP_FRAME_WIDTH: 2,
            cv2.CAP_PROP_FRAME_HEIGHT: 1,
            cv2.CAP_PROP_FPS: 30,
        }.get(property_id, 0)

    def read(self):
        self.read_count += 1
        if self.fail_after_first and self.read_count > 1:
            time.sleep(0.002)
            return False, None
        time.sleep(0.002)
        return True, np.array([[[1, 2, 3], [10, 20, 30]]], dtype=np.uint8)

    def release(self):
        self.released = True
        self.opened = False


def device(name="Integrated Camera", path="camera-path", backend=cv2.CAP_MSMF):
    return SimpleNamespace(index=0, name=name, path=path, backend=backend)


def wait_for_frame(source, minimum_id=0):
    deadline = time.monotonic() + 1
    while time.monotonic() < deadline:
        frame = source.read_latest()
        if frame is not None and frame.frame_id >= minimum_id:
            return frame
        time.sleep(0.002)
    raise AssertionError("Timed out waiting for a frame")


def test_source_enumerates_named_devices_and_marks_virtual_inputs():
    items = [device(), device("OBS Virtual Camera", "virtual-path")]
    source = OpenCVCameraSource(enumerator=lambda _backend: items)

    devices = source.enumerate_devices()

    assert [item.display_name for item in devices] == ["Integrated Camera", "OBS Virtual Camera"]
    assert [item.is_virtual for item in devices] == [False, True]
    assert devices[0].device_id != devices[1].device_id


def test_source_publishes_timestamped_immutable_rgb_latest_frames():
    captures = []
    ticks = itertools.count(1_000, 10)

    def capture_factory(index, backend):
        capture = FakeCapture(index, backend)
        captures.append(capture)
        return capture

    source = OpenCVCameraSource(
        enumerator=lambda _backend: [device()],
        capture_factory=capture_factory,
        monotonic_ns=lambda: next(ticks),
    )
    camera = source.enumerate_devices()[0]
    negotiated = source.open(camera.device_id, FrameFormat(1280, 720, 30))
    frame = wait_for_frame(source)

    assert negotiated == FrameFormat(2, 1, 30)
    assert frame.timestamp_ns >= 1_000
    assert frame.rgb.flags.c_contiguous
    assert not frame.rgb.flags.writeable
    assert frame.rgb.tolist() == [[[3, 2, 1], [30, 20, 10]]]
    assert captures[0].properties[cv2.CAP_PROP_FRAME_WIDTH] == 1280
    assert captures[0].properties[cv2.CAP_PROP_FRAME_HEIGHT] == 720
    source.close()
    source.close()
    assert captures[0].released


def test_source_rejects_virtual_or_missing_camera():
    source = OpenCVCameraSource(enumerator=lambda _backend: [device("Virtual Camera")])
    virtual = source.enumerate_devices()[0]
    with pytest.raises(StageError, match="physical camera"):
        source.open(virtual.device_id, FrameFormat(1280, 720, 30))
    with pytest.raises(StageError, match="no longer available"):
        source.open("missing", FrameFormat(1280, 720, 30))


def test_source_reports_disconnect_after_bounded_failure_period():
    source = OpenCVCameraSource(
        disconnect_timeout_seconds=0.01,
        enumerator=lambda _backend: [device()],
        capture_factory=lambda index, backend: FakeCapture(index, backend, fail_after_first=True),
    )
    camera = source.enumerate_devices()[0]
    source.open(camera.device_id, FrameFormat(1280, 720, 30))

    deadline = time.monotonic() + 1
    while time.monotonic() < deadline:
        try:
            source.read_latest()
        except StageError as exc:
            assert "Lost video" in str(exc)
            break
        time.sleep(0.005)
    else:
        raise AssertionError("Disconnect was not reported")
    source.close()


@pytest.mark.parametrize(
    "frame_format",
    [
        (0, 720, 30),
        (1280, 0, 30),
        (1280, 720, 0),
        (1280, 720, float("nan")),
        (1280, 720, float("inf")),
        (True, 720, 30),
    ],
)
def test_invalid_frame_format_is_rejected(frame_format):
    with pytest.raises(ValueError):
        FrameFormat(*frame_format)
