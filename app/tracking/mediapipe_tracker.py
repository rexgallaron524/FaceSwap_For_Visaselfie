"""Single-face MediaPipe live-stream adapter with bounded asynchronous state."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any, Protocol

from app.pipeline.types import (
    FaceState,
    NormalizedLandmark,
    Point2D,
    StageError,
    TrackingResult,
    TrackingStatus,
    VideoFrame,
)
from app.reference.detector import default_model_path
from app.tracking.geometry import (
    canonical_blendshape_name,
    derived_tracking_confidence,
    landmark_bounds,
    normalized_to_pixel,
    rotation_matrix_to_head_pose,
)

LANDMARK_SCHEMA = "mediapipe-face-landmarker-478-v1"


class _LiveBackend(Protocol):
    def submit(self, rgb, timestamp_ms: int) -> None: ...

    def close(self) -> None: ...


class _MediaPipeLiveBackend:
    def __init__(self, model_path: Path, callback: Callable[[Any, Any, int], None]) -> None:
        import mediapipe as mp

        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            result_callback=callback,
        )
        self._mp = mp
        self._landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)

    def submit(self, rgb, timestamp_ms: int) -> None:
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        self._landmarker.detect_async(image, timestamp_ms)

    def close(self) -> None:
        self._landmarker.close()


@dataclass(frozen=True, slots=True)
class _PendingFrame:
    frame_id: int
    timestamp_ns: int
    timestamp_ms: int
    width: int
    height: int
    submitted_at_ns: int


class MediaPipeFaceTracker:
    """At most one frame is in flight; busy submissions are explicitly dropped."""

    def __init__(
        self,
        model_path: Path | None = None,
        *,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
        backend_factory: Callable[[Path, Callable[[Any, Any, int], None]], _LiveBackend]
        | None = None,
    ) -> None:
        self._model_path = (model_path or default_model_path()).resolve()
        self._monotonic_ns = monotonic_ns
        self._backend_factory = backend_factory or _MediaPipeLiveBackend
        self._lock = threading.Lock()
        self._backend: _LiveBackend | None = None
        self._pending: _PendingFrame | None = None
        self._latest: TrackingResult | None = None
        self._last_frame_id = -1
        self._last_timestamp_ns = -1
        self._last_timestamp_ms = -1
        self._accept_callbacks = False

    def open(self) -> None:
        with self._lock:
            if self._backend is not None:
                return
        if not self._model_path.is_file():
            raise StageError(f"Face Landmarker model is missing: {self._model_path}")
        try:
            backend = self._backend_factory(self._model_path, self._on_result)
        except Exception as exc:
            raise StageError(f"Could not initialize live Face Landmarker: {exc}") from exc
        with self._lock:
            self._backend = backend
            self._pending = None
            self._latest = None
            self._last_frame_id = -1
            self._last_timestamp_ns = -1
            self._last_timestamp_ms = -1
            self._accept_callbacks = True

    def submit(self, frame: VideoFrame) -> bool:
        if frame.rgb.ndim != 3 or frame.rgb.shape[2] != 3:
            raise StageError("Face tracking requires an HxWx3 RGB frame")
        height, width = frame.rgb.shape[:2]
        if width <= 0 or height <= 0:
            raise StageError("Face tracking requires a nonempty RGB frame")
        with self._lock:
            backend = self._backend
            if backend is None:
                raise StageError("Face tracker is not open")
            if (
                frame.frame_id <= self._last_frame_id
                or frame.timestamp_ns <= self._last_timestamp_ns
            ):
                raise StageError("Tracking frames must have increasing IDs and timestamps")
            self._last_frame_id = frame.frame_id
            self._last_timestamp_ns = frame.timestamp_ns
            # A completed result still owns its retained input until poll_latest consumes it.
            # Accepting another frame here could replace the UI's one matching frame before
            # the completed result is observed.
            if self._pending is not None or self._latest is not None:
                return False
            timestamp_ms = max(frame.timestamp_ns // 1_000_000, self._last_timestamp_ms + 1)
            self._last_timestamp_ms = timestamp_ms
            pending = _PendingFrame(
                frame.frame_id,
                frame.timestamp_ns,
                timestamp_ms,
                width,
                height,
                self._monotonic_ns(),
            )
            self._pending = pending
        try:
            backend.submit(frame.rgb, timestamp_ms)
        except Exception as exc:
            with self._lock:
                if self._pending is pending:
                    self._pending = None
            raise StageError(f"Could not submit frame to Face Landmarker: {exc}") from exc
        return True

    def poll_latest(self) -> TrackingResult | None:
        with self._lock:
            result = self._latest
            self._latest = None
            return result

    def _on_result(self, result: Any, _output_image: Any, timestamp_ms: int) -> None:
        with self._lock:
            pending = self._pending
            accepting = self._accept_callbacks
        if not accepting or pending is None:
            return
        latency_ms = max(0.0, (self._monotonic_ns() - pending.submitted_at_ns) / 1_000_000)
        try:
            if timestamp_ms != pending.timestamp_ms:
                raise ValueError("Face Landmarker returned an unexpected timestamp")
            converted = self._convert_result(result, pending, latency_ms)
        except Exception as exc:
            converted = TrackingResult(
                pending.frame_id,
                pending.timestamp_ns,
                TrackingStatus.ERROR,
                error=f"Face tracking result was invalid: {exc}",
                latency_ms=latency_ms,
            )
        with self._lock:
            if self._accept_callbacks and self._pending is pending:
                self._latest = converted
                self._pending = None

    @staticmethod
    def _convert_result(result: Any, pending: _PendingFrame, latency_ms: float) -> TrackingResult:
        faces = result.face_landmarks or []
        if not faces:
            return TrackingResult(
                pending.frame_id,
                pending.timestamp_ns,
                TrackingStatus.NO_FACE,
                latency_ms=latency_ms,
            )
        if len(faces) != 1:
            raise ValueError("single-face tracker returned multiple faces")
        normalized = tuple(
            NormalizedLandmark(float(point.x), float(point.y), float(point.z)) for point in faces[0]
        )
        if len(normalized) != 478:
            raise ValueError(f"expected 478 landmarks, received {len(normalized)}")
        bounds = landmark_bounds(normalized, pending.width, pending.height)
        if bounds.width <= 0 or bounds.height <= 0:
            raise ValueError("face bounds are empty")
        matrices = result.facial_transformation_matrixes or []
        if len(matrices) != 1:
            raise ValueError("facial transformation matrix is missing")
        pose = rotation_matrix_to_head_pose(matrices[0])
        pixels = tuple(
            normalized_to_pixel(point, pending.width, pending.height) for point in normalized
        )
        blendshapes: dict[str, float] = {}
        shape_lists = result.face_blendshapes or []
        if shape_lists:
            for category in shape_lists[0]:
                name = category.category_name or category.display_name
                score = float(category.score)
                if name and isfinite(score):
                    blendshapes[canonical_blendshape_name(str(name))] = max(0.0, min(1.0, score))
        face = FaceState(
            frame_id=pending.frame_id,
            timestamp_ns=pending.timestamp_ns,
            bounds=bounds,
            center=Point2D(bounds.x + bounds.width / 2, bounds.y + bounds.height / 2),
            scale=bounds.width / pending.width,
            pose=pose,
            landmarks=pixels,
            normalized_landmarks=normalized,
            landmark_schema=LANDMARK_SCHEMA,
            tracking_confidence=derived_tracking_confidence(normalized),
            blendshapes=blendshapes,
        )
        return TrackingResult(
            pending.frame_id,
            pending.timestamp_ns,
            TrackingStatus.TRACKED,
            face=face,
            latency_ms=latency_ms,
        )

    def close(self) -> None:
        with self._lock:
            backend = self._backend
            self._backend = None
            self._accept_callbacks = False
            self._pending = None
            self._latest = None
        if backend is not None:
            try:
                backend.close()
            except Exception as exc:
                raise StageError(f"Could not close Face Landmarker: {exc}") from exc
