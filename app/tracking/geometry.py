"""Backend-independent coordinate, pose, and tracking-quality helpers."""

from __future__ import annotations

import re
from math import asin, atan2, degrees, hypot, isfinite, sqrt
from typing import Any

import numpy as np

from app.pipeline.types import HeadPose, NormalizedLandmark, Point2D, Rect

_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def normalized_to_pixel(point: NormalizedLandmark, width: int, height: int) -> Point2D:
    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive")
    return Point2D(point.x * width, point.y * height)


def landmark_bounds(landmarks: tuple[NormalizedLandmark, ...], width: int, height: int) -> Rect:
    if not landmarks:
        raise ValueError("At least one landmark is required")
    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive")
    if not all(isfinite(value) for point in landmarks for value in (point.x, point.y, point.z)):
        raise ValueError("Landmarks must be finite")
    xs = [point.x for point in landmarks]
    ys = [point.y for point in landmarks]
    return Rect(
        min(xs) * width,
        min(ys) * height,
        (max(xs) - min(xs)) * width,
        (max(ys) - min(ys)) * height,
    )


def rotation_matrix_to_head_pose(value: Any) -> HeadPose:
    """Convert a canonical-face rotation to +right yaw, +down pitch, +clockwise roll."""
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape not in ((3, 3), (4, 4)):
        raise ValueError("Facial transformation matrix must be 3×3 or 4×4")
    rotation = matrix[:3, :3]
    if not np.isfinite(rotation).all():
        raise ValueError("Facial transformation matrix must be finite")

    # MediaPipe matrices can contain small scale/shear errors. Project to the nearest
    # proper rotation before extracting Rz(roll) * Ry(yaw) * Rx(pitch) Euler angles.
    left, _, right = np.linalg.svd(rotation)
    rotation = left @ right
    if np.linalg.det(rotation) < 0:
        left[:, -1] *= -1
        rotation = left @ right

    horizontal = hypot(rotation[0, 0], rotation[1, 0])
    if horizontal > 1e-7:
        pitch = atan2(rotation[2, 1], rotation[2, 2])
        yaw = atan2(-rotation[2, 0], horizontal)
        roll = atan2(rotation[1, 0], rotation[0, 0])
    else:
        pitch = atan2(-rotation[1, 2], rotation[1, 1])
        yaw = asin(max(-1.0, min(1.0, -rotation[2, 0])))
        roll = 0.0
    return HeadPose(degrees(yaw), degrees(pitch), degrees(roll))


def derived_tracking_confidence(landmarks: tuple[NormalizedLandmark, ...]) -> float:
    """Return a documented quality heuristic when the backend exposes no confidence."""
    if not landmarks:
        return 0.0
    if not all(isfinite(value) for point in landmarks for value in (point.x, point.y, point.z)):
        raise ValueError("Landmarks must be finite")
    in_frame = sum(0 <= point.x <= 1 and 0 <= point.y <= 1 for point in landmarks) / len(landmarks)
    width = max(point.x for point in landmarks) - min(point.x for point in landmarks)
    height = max(point.y for point in landmarks) - min(point.y for point in landmarks)
    size_quality = max(0.0, min(1.0, max(width, height) / 0.18))
    return max(0.0, min(1.0, sqrt(in_frame * size_quality)))


def canonical_blendshape_name(name: str) -> str:
    return _CAMEL_BOUNDARY.sub("_", name).replace("-", "_").casefold()
