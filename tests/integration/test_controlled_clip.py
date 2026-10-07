from pathlib import Path

import cv2


def test_controlled_motion_clip_is_readable_and_has_documented_timing():
    path = Path(__file__).resolve().parents[1] / "assets" / "controlled_motion.mp4"
    capture = cv2.VideoCapture(str(path))
    try:
        assert capture.isOpened()
        assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 240
        assert capture.get(cv2.CAP_PROP_FPS) == 30.0
        assert int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) == 640
        assert int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) == 360
        ok, frame = capture.read()
        assert ok
        assert frame.shape == (360, 640, 3)
    finally:
        capture.release()
