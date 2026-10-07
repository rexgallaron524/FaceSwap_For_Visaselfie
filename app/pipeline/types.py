"""Backend-independent data contracts; see docs/architecture.md for units/ownership."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from math import isfinite
from types import MappingProxyType

import numpy as np
from numpy.typing import NDArray

type RGBImage = NDArray[np.uint8]
type AlphaMask = NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class FrameFormat:
    width: int
    height: int
    fps: float

    def __post_init__(self) -> None:
        if type(self.width) is not int or self.width <= 0:
            raise ValueError("Frame width must be a positive integer")
        if type(self.height) is not int or self.height <= 0:
            raise ValueError("Frame height must be a positive integer")
        if (
            isinstance(self.fps, bool)
            or not isinstance(self.fps, (int, float))
            or not isfinite(self.fps)
            or self.fps <= 0
        ):
            raise ValueError("Frame FPS must be positive")


@dataclass(frozen=True, slots=True)
class VideoFrame:
    """C-contiguous H×W×3 RGB uint8; capture timestamp uses monotonic_ns()."""

    frame_id: int
    timestamp_ns: int
    rgb: RGBImage


@dataclass(frozen=True, slots=True)
class CameraDevice:
    device_id: str
    display_name: str
    is_virtual: bool


@dataclass(frozen=True, slots=True)
class Point2D:
    x: float
    y: float


@dataclass(frozen=True, slots=True)
class NormalizedLandmark:
    """Backend-neutral landmark: normalized image x/y and face-width-relative z."""

    x: float
    y: float
    z: float = 0.0


@dataclass(frozen=True, slots=True)
class Rect:
    x: float
    y: float
    width: float
    height: float


@dataclass(frozen=True, slots=True)
class HeadPose:
    """Degrees: +yaw toward image right, +pitch down, +roll clockwise."""

    yaw: float
    pitch: float
    roll: float


@dataclass(frozen=True, slots=True)
class FaceState:
    frame_id: int
    timestamp_ns: int
    bounds: Rect
    center: Point2D
    scale: float  # Face bounding width / full image width.
    pose: HeadPose
    landmarks: tuple[Point2D, ...]  # Full-frame pixels; topology identified below.
    normalized_landmarks: tuple[NormalizedLandmark, ...]
    landmark_schema: str  # Versioned adapter-defined ordering, never inferred by count.
    tracking_confidence: float  # [0, 1]; adapter must document its derivation.
    blendshapes: Mapping[str, float] = field(default_factory=dict)  # [0, 1] coefficients.

    def __post_init__(self) -> None:
        if type(self.frame_id) is not int or self.frame_id < 0:
            raise ValueError("FaceState frame ID must be a nonnegative integer")
        if type(self.timestamp_ns) is not int or self.timestamp_ns < 0:
            raise ValueError("FaceState timestamp must be a nonnegative integer")
        numeric_values = (
            self.bounds.x,
            self.bounds.y,
            self.bounds.width,
            self.bounds.height,
            self.center.x,
            self.center.y,
            self.scale,
            self.pose.yaw,
            self.pose.pitch,
            self.pose.roll,
            self.tracking_confidence,
        )
        if not all(isfinite(value) for value in numeric_values):
            raise ValueError("FaceState geometry and confidence must be finite")
        if self.bounds.width <= 0 or self.bounds.height <= 0 or self.scale <= 0:
            raise ValueError("FaceState bounds and scale must be positive")
        if not 0 <= self.tracking_confidence <= 1:
            raise ValueError("FaceState tracking confidence must be in [0, 1]")
        if not self.landmarks or len(self.landmarks) != len(self.normalized_landmarks):
            raise ValueError("FaceState pixel and normalized landmarks must be nonempty and match")
        if not all(isfinite(value) for point in self.landmarks for value in (point.x, point.y)):
            raise ValueError("FaceState pixel landmarks must be finite")
        if not all(
            isfinite(value)
            for point in self.normalized_landmarks
            for value in (point.x, point.y, point.z)
        ):
            raise ValueError("FaceState normalized landmarks must be finite")
        if not self.landmark_schema:
            raise ValueError("FaceState landmark schema is required")
        if not isinstance(self.blendshapes, Mapping):
            raise ValueError("FaceState blendshapes must be a mapping")
        if not all(
            isfinite(float(value)) and 0 <= float(value) <= 1 for value in self.blendshapes.values()
        ):
            raise ValueError("FaceState blendshapes must be finite values in [0, 1]")
        if not isinstance(self.blendshapes, MappingProxyType):
            object.__setattr__(self, "blendshapes", MappingProxyType(dict(self.blendshapes)))


class TrackingStatus(Enum):
    TRACKED = "tracked"
    NO_FACE = "no_face"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class TrackingResult:
    frame_id: int
    timestamp_ns: int
    status: TrackingStatus
    face: FaceState | None = None
    error: str | None = None
    latency_ms: float | None = None

    def __post_init__(self) -> None:
        if type(self.frame_id) is not int or self.frame_id < 0:
            raise ValueError("TrackingResult frame ID must be a nonnegative integer")
        if type(self.timestamp_ns) is not int or self.timestamp_ns < 0:
            raise ValueError("TrackingResult timestamp must be a nonnegative integer")
        if self.latency_ms is not None and (not isfinite(self.latency_ms) or self.latency_ms < 0):
            raise ValueError("Tracking latency must be finite and nonnegative")
        if self.status is TrackingStatus.TRACKED:
            if self.face is None or self.error is not None:
                raise ValueError("TRACKED requires a face and no error")
            if (self.face.frame_id, self.face.timestamp_ns) != (
                self.frame_id,
                self.timestamp_ns,
            ):
                raise ValueError("TrackingResult and FaceState identities must match")
        elif self.status is TrackingStatus.NO_FACE:
            if self.face is not None or self.error is not None:
                raise ValueError("NO_FACE cannot carry a face or error")
        elif self.face is not None or not self.error:
            raise ValueError("ERROR requires an error and no face")


@dataclass(frozen=True, slots=True)
class PreparedReference:
    reference_id: str
    yaw: float
    pitch: float
    expression: str
    rgb: RGBImage  # Aligned/cached face image, owned by the library.
    landmarks: tuple[Point2D, ...]  # Pixels in this aligned reference image.
    landmark_schema: str


@dataclass(frozen=True, slots=True)
class ReferenceWeight:
    reference_id: str
    weight: float  # Finite, nonnegative; a nonempty selection sums to 1.

    def __post_init__(self) -> None:
        if not self.reference_id:
            raise ValueError("Reference weight ID is required")
        if not isfinite(self.weight) or self.weight < 0:
            raise ValueError("Reference weight must be finite and nonnegative")


@dataclass(frozen=True, slots=True)
class RenderedFace:
    frame_id: int
    timestamp_ns: int
    rgb: RGBImage  # Full output frame size; content outside alpha is irrelevant.
    alpha: AlphaMask  # H×W, finite [0, 1], zero outside the facial region.


class StageError(RuntimeError):
    """Recoverable stage failure; orchestration must display/emit a placeholder."""
