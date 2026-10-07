"""Backend-independent data contracts; see docs/architecture.md for units/ownership."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from numpy.typing import NDArray

type RGBImage = NDArray[np.uint8]
type AlphaMask = NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class FrameFormat:
    width: int
    height: int
    fps: float


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
    landmark_schema: str  # Versioned adapter-defined ordering, never inferred by count.
    tracking_confidence: float  # [0, 1]; adapter must document its derivation.
    blendshapes: Mapping[str, float] = field(default_factory=dict)  # [0, 1] coefficients.


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


@dataclass(frozen=True, slots=True)
class RenderedFace:
    frame_id: int
    timestamp_ns: int
    rgb: RGBImage  # Full output frame size; content outside alpha is irrelevant.
    alpha: AlphaMask  # H×W, finite [0, 1], zero outside the facial region.


class StageError(RuntimeError):
    """Recoverable stage failure; orchestration must display/emit a placeholder."""
