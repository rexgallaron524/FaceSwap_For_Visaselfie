from pathlib import Path

import cv2
import numpy as np
import pytest

from app.reference.detector import FaceObservation, NormalizedLandmark
from app.reference.preprocessor import ReferencePreprocessor, ReferenceValidationError, initial_slot


class FakeDetector:
    def __init__(self, observations):
        self.observations = observations
        self.closed = False

    def detect(self, _rgb):
        return self.observations

    def close(self):
        self.closed = True


def observation() -> FaceObservation:
    points = []
    for index in range(478):
        angle = 2 * np.pi * index / 478
        points.append(NormalizedLandmark(0.5 + 0.22 * np.cos(angle), 0.5 + 0.3 * np.sin(angle)))
    for index in (33, 133, 159, 145, 468, 469, 470, 471, 472):
        points[index] = NormalizedLandmark(0.4, 0.42)
    for index in (362, 263, 386, 374, 473, 474, 475, 476, 477):
        points[index] = NormalizedLandmark(0.6, 0.42)
    return FaceObservation(tuple(points), {"mouthSmileLeft": 0.2})


def write_sharp_image(path: Path) -> None:
    grid = np.indices((512, 512)).sum(axis=0) % 2
    gray = (grid * 150 + 50).astype(np.uint8)
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    assert cv2.imwrite(str(path), bgr)


def test_preprocessor_aligns_normalizes_and_caches_landmarks(tmp_path):
    image_path = tmp_path / "front.png"
    write_sharp_image(image_path)
    detector = FakeDetector((observation(),))
    processor = ReferencePreprocessor(detector, clock=lambda: "2026-10-07T00:00:00+00:00")

    result = processor.process(initial_slot("front-neutral"), image_path)

    assert result.rgb.shape == (512, 512, 3)
    assert result.rgb.dtype == np.uint8
    assert result.rgb.flags.c_contiguous
    assert not result.rgb.flags.writeable
    assert len(result.landmarks) == 478
    assert result.metadata.slot_id == "front-neutral"
    assert result.metadata.source_name == "front.png"
    assert result.metadata.original_width == 512
    assert result.metadata.created_at == "2026-10-07T00:00:00+00:00"
    assert result.metadata.sharpness > 18
    assert result.metadata.blendshapes["mouthSmileLeft"] == 0.2
    processor.close()
    assert detector.closed


@pytest.mark.parametrize(
    ("observations", "message"),
    [
        ((), "No usable face"),
        ((observation(), observation()), "exactly one face"),
    ],
)
def test_preprocessor_rejects_missing_or_multiple_faces(tmp_path, observations, message):
    image_path = tmp_path / "reference.png"
    write_sharp_image(image_path)
    processor = ReferencePreprocessor(FakeDetector(observations))

    with pytest.raises(ReferenceValidationError, match=message):
        processor.process(initial_slot("front-neutral"), image_path)


def test_preprocessor_rejects_unreadable_and_small_images(tmp_path):
    processor = ReferencePreprocessor(FakeDetector((observation(),)))
    broken = tmp_path / "broken.jpg"
    broken.write_bytes(b"not an image")
    with pytest.raises(ReferenceValidationError, match="not a readable image"):
        processor.process(initial_slot("front-neutral"), broken)

    small = tmp_path / "small.png"
    assert cv2.imwrite(str(small), np.zeros((100, 100, 3), np.uint8))
    with pytest.raises(ReferenceValidationError, match="at least 256"):
        processor.process(initial_slot("front-neutral"), small)
