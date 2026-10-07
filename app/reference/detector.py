"""Face-landmark detection boundary and MediaPipe image-mode adapter."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Protocol

from app.pipeline.types import RGBImage, StageError


@dataclass(frozen=True, slots=True)
class NormalizedLandmark:
    x: float
    y: float
    z: float = 0.0


@dataclass(frozen=True, slots=True)
class FaceObservation:
    landmarks: tuple[NormalizedLandmark, ...]
    blendshapes: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.blendshapes, MappingProxyType):
            object.__setattr__(self, "blendshapes", MappingProxyType(dict(self.blendshapes)))


class ReferenceFaceDetector(Protocol):
    def detect(self, rgb: RGBImage) -> tuple[FaceObservation, ...]: ...

    def close(self) -> None: ...


def default_model_path() -> Path:
    packaged = Path(__file__).resolve().parent / "assets" / "face_landmarker.task"
    if packaged.is_file():
        return packaged
    return Path(__file__).resolve().parents[2] / "models" / "face_landmarker.task"


class MediaPipeReferenceFaceDetector:
    """Lazy MediaPipe Face Landmarker configured for still enrollment images."""

    def __init__(self, model_path: Path | None = None) -> None:
        self._model_path = (model_path or default_model_path()).resolve()
        self._landmarker = None
        self._mp = None

    def _ensure_open(self) -> None:
        if self._landmarker is not None:
            return
        if not self._model_path.is_file():
            raise StageError(f"Face Landmarker model is missing: {self._model_path}")
        try:
            import mediapipe as mp

            options = mp.tasks.vision.FaceLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=str(self._model_path)),
                running_mode=mp.tasks.vision.RunningMode.IMAGE,
                num_faces=2,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                output_face_blendshapes=True,
            )
            self._landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)
            self._mp = mp
        except Exception as exc:
            raise StageError(f"Could not initialize Face Landmarker: {exc}") from exc

    def detect(self, rgb: RGBImage) -> tuple[FaceObservation, ...]:
        self._ensure_open()
        try:
            image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
            result = self._landmarker.detect(image)
        except Exception as exc:
            raise StageError(f"Face detection failed: {exc}") from exc

        observations: list[FaceObservation] = []
        shape_lists = result.face_blendshapes or []
        for index, landmarks in enumerate(result.face_landmarks):
            blendshapes: dict[str, float] = {}
            if index < len(shape_lists):
                for category in shape_lists[index]:
                    name = category.category_name or category.display_name
                    if name:
                        blendshapes[str(name)] = float(category.score)
            observations.append(
                FaceObservation(
                    landmarks=tuple(
                        NormalizedLandmark(float(point.x), float(point.y), float(point.z))
                        for point in landmarks
                    ),
                    blendshapes=blendshapes,
                )
            )
        return tuple(observations)

    def close(self) -> None:
        if self._landmarker is not None:
            self._landmarker.close()
        self._landmarker = None
        self._mp = None
