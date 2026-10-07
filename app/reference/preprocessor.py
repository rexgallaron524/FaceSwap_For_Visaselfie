"""One-time validation, alignment, and normalization for enrollment images."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from datetime import UTC, datetime
from math import atan2, degrees, hypot, isfinite
from pathlib import Path

import cv2
import numpy as np

from app.pipeline.types import Point2D, Rect, StageError
from app.reference.detector import MediaPipeReferenceFaceDetector, ReferenceFaceDetector
from app.reference.model import (
    INITIAL_REFERENCE_SLOTS,
    NORMALIZED_FACE_SIZE,
    ProcessedReference,
    ReferenceMetadata,
    ReferenceSlot,
)

_LEFT_EYE = (33, 133, 159, 145, 468, 469, 470, 471, 472)
_RIGHT_EYE = (362, 263, 386, 374, 473, 474, 475, 476, 477)


class ReferenceValidationError(StageError):
    """An enrollment image is readable but not suitable as a reference."""


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


class ReferencePreprocessor:
    def __init__(
        self,
        detector: ReferenceFaceDetector | None = None,
        *,
        clock: Callable[[], str] = _utc_now,
    ) -> None:
        self._detector = detector or MediaPipeReferenceFaceDetector()
        self._clock = clock
        self._size = NORMALIZED_FACE_SIZE

    def process(self, slot: ReferenceSlot, path: Path) -> ProcessedReference:
        try:
            encoded = path.read_bytes()
        except OSError as exc:
            raise ReferenceValidationError(f"Could not read image: {exc}") from exc
        if not encoded:
            raise ReferenceValidationError("The selected image is empty")
        if len(encoded) > 30 * 1024 * 1024:
            raise ReferenceValidationError("Reference images must be 30 MB or smaller")

        decoded = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if decoded is None:
            raise ReferenceValidationError("The selected file is not a readable image")
        rgb = self._to_rgb(decoded)
        height, width = rgb.shape[:2]
        if width < 256 or height < 256:
            raise ReferenceValidationError("Reference images must be at least 256 × 256 pixels")
        if width > 8192 or height > 8192:
            raise ReferenceValidationError("Reference images must be 8192 pixels or smaller")

        observations = self._detector.detect(rgb)
        if not observations:
            raise ReferenceValidationError("No usable face was detected")
        if len(observations) > 1:
            raise ReferenceValidationError("Use an image containing exactly one face")
        observation = observations[0]
        if len(observation.landmarks) < 478:
            raise ReferenceValidationError("The complete facial landmark mesh was not detected")
        if not all(
            isfinite(value)
            for point in observation.landmarks
            for value in (point.x, point.y, point.z)
        ):
            raise ReferenceValidationError("The face detector returned invalid landmarks")

        xs = [point.x for point in observation.landmarks]
        ys = [point.y for point in observation.landmarks]
        x0, y0 = max(0.0, min(xs)), max(0.0, min(ys))
        x1, y1 = min(1.0, max(xs)), min(1.0, max(ys))
        box_width, box_height = x1 - x0, y1 - y0
        face_coverage = box_width * box_height
        if box_width * width < 96 or box_height * height < 96 or face_coverage < 0.025:
            raise ReferenceValidationError(
                "The detected face is too small; move closer to the camera"
            )

        crop = rgb[
            int(y0 * height) : max(int(y1 * height), int(y0 * height) + 1),
            int(x0 * width) : max(int(x1 * width), int(x0 * width) + 1),
        ]
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        brightness = float(np.mean(gray))
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if brightness < 25:
            raise ReferenceValidationError("The face is too dark; use brighter, even lighting")
        if brightness > 240:
            raise ReferenceValidationError("The face is overexposed; reduce direct light")
        if sharpness < 18:
            raise ReferenceValidationError("The face is too blurry; use a sharper image")

        left_eye = self._landmark_center(observation.landmarks, _LEFT_EYE, width, height)
        right_eye = self._landmark_center(observation.landmarks, _RIGHT_EYE, width, height)
        if left_eye[0] > right_eye[0]:
            left_eye, right_eye = right_eye, left_eye
        eye_dx, eye_dy = right_eye[0] - left_eye[0], right_eye[1] - left_eye[1]
        eye_distance = hypot(eye_dx, eye_dy)
        if eye_distance < 24:
            raise ReferenceValidationError("Eye landmarks are too close for reliable alignment")

        transform, rotation = self._alignment_transform(left_eye, right_eye)
        normalized = cv2.warpAffine(
            rgb,
            transform,
            (self._size, self._size),
            flags=cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        normalized = np.ascontiguousarray(normalized, dtype=np.uint8)
        normalized.setflags(write=False)
        landmarks = tuple(
            Point2D(
                transform[0, 0] * point.x * width
                + transform[0, 1] * point.y * height
                + transform[0, 2],
                transform[1, 0] * point.x * width
                + transform[1, 1] * point.y * height
                + transform[1, 2],
            )
            for point in observation.landmarks
        )
        metadata = ReferenceMetadata(
            slot_id=slot.slot_id,
            source_name=path.name,
            source_sha256=hashlib.sha256(encoded).hexdigest(),
            original_width=width,
            original_height=height,
            face_bounds=Rect(x0, y0, box_width, box_height),
            brightness=round(brightness, 3),
            sharpness=round(sharpness, 3),
            face_coverage=round(face_coverage, 6),
            alignment_rotation_degrees=round(rotation, 3),
            created_at=self._clock(),
            blendshapes=observation.blendshapes,
        )
        return ProcessedReference(metadata, normalized, landmarks)

    @staticmethod
    def _to_rgb(decoded: np.ndarray) -> np.ndarray:
        if decoded.ndim == 2:
            return np.ascontiguousarray(cv2.cvtColor(decoded, cv2.COLOR_GRAY2RGB))
        if decoded.ndim != 3:
            raise ReferenceValidationError("The image has an unsupported pixel format")
        if decoded.shape[2] == 3:
            return np.ascontiguousarray(cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB))
        if decoded.shape[2] == 4:
            return np.ascontiguousarray(cv2.cvtColor(decoded, cv2.COLOR_BGRA2RGB))
        raise ReferenceValidationError("The image has an unsupported channel layout")

    @staticmethod
    def _landmark_center(landmarks, indices, width: int, height: int) -> tuple[float, float]:
        return (
            sum(landmarks[index].x for index in indices) * width / len(indices),
            sum(landmarks[index].y for index in indices) * height / len(indices),
        )

    def _alignment_transform(
        self, left_eye: tuple[float, float], right_eye: tuple[float, float]
    ) -> tuple[np.ndarray, float]:
        dx, dy = right_eye[0] - left_eye[0], right_eye[1] - left_eye[1]
        source_distance = hypot(dx, dy)
        target_left = (self._size * 0.32, self._size * 0.38)
        target_right = (self._size * 0.68, self._size * 0.38)
        target_distance = target_right[0] - target_left[0]
        scale = target_distance / source_distance
        cosine, sine = dx / source_distance, dy / source_distance
        a, b = scale * cosine, scale * sine
        source_mid = ((left_eye[0] + right_eye[0]) / 2, (left_eye[1] + right_eye[1]) / 2)
        target_mid = ((target_left[0] + target_right[0]) / 2, target_left[1])
        transform = np.array(
            [
                [a, b, target_mid[0] - a * source_mid[0] - b * source_mid[1]],
                [-b, a, target_mid[1] + b * source_mid[0] - a * source_mid[1]],
            ],
            dtype=np.float64,
        )
        return transform, degrees(atan2(dy, dx))

    def close(self) -> None:
        self._detector.close()


def initial_slot(slot_id: str) -> ReferenceSlot:
    for slot in INITIAL_REFERENCE_SLOTS:
        if slot.slot_id == slot_id:
            return slot
    raise KeyError(slot_id)
