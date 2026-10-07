"""Deterministic landmark-guided face renderer for the geometric prototype."""

from __future__ import annotations

from math import isfinite

import cv2
import numpy as np

from app.pipeline.types import (
    FaceState,
    FrameFormat,
    PreparedReference,
    ReferenceWeight,
    RenderedFace,
    StageError,
)

# MediaPipe Face Landmarker v1 indices. The renderer checks the schema before using
# them so this backend detail does not leak into the rest of the pipeline.
_LANDMARK_SCHEMA = "mediapipe-face-landmarker-478-v1"
_FACE_OVAL = (
    10,
    338,
    297,
    332,
    284,
    251,
    389,
    356,
    454,
    323,
    361,
    288,
    397,
    365,
    379,
    378,
    400,
    377,
    152,
    148,
    176,
    149,
    150,
    136,
    172,
    58,
    132,
    93,
    234,
    127,
    162,
    21,
    54,
    103,
    67,
    109,
)
_FEATURE_CONTOURS = (
    33,
    7,
    163,
    144,
    145,
    153,
    154,
    155,
    133,
    173,
    157,
    158,
    159,
    160,
    161,
    246,
    362,
    382,
    381,
    380,
    374,
    373,
    390,
    249,
    263,
    466,
    388,
    387,
    386,
    385,
    384,
    398,
    70,
    63,
    105,
    66,
    107,
    336,
    296,
    334,
    293,
    300,
    1,
    2,
    4,
    5,
    45,
    98,
    168,
    195,
    197,
    275,
    327,
    61,
    146,
    91,
    181,
    84,
    17,
    314,
    405,
    321,
    375,
    291,
    308,
    324,
    318,
    402,
    317,
    14,
    87,
    178,
    88,
    95,
)
_MESH_INDICES = tuple(sorted(set(range(0, 468, 6)) | set(_FACE_OVAL) | set(_FEATURE_CONTOURS)))
_WEIGHT_EPSILON = 1e-8


def delaunay_triangle_indices(
    points: np.ndarray, width: int, height: int
) -> tuple[tuple[int, int, int], ...]:
    """Return stable Delaunay triangles referring to the original point indices."""
    values = np.asarray(points, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("Delaunay points must be an N x 2 array")
    if width <= 0 or height <= 0:
        raise ValueError("Delaunay bounds must be positive")
    if not np.isfinite(values).all():
        raise ValueError("Delaunay points must be finite")

    retained: list[tuple[int, np.ndarray]] = []
    occupied: set[tuple[int, int]] = set()
    for index, point in enumerate(values):
        if not (0.0 <= point[0] < width and 0.0 <= point[1] < height):
            continue
        key = (round(float(point[0]) * 100), round(float(point[1]) * 100))
        if key in occupied:
            continue
        occupied.add(key)
        retained.append((index, point))
    if len(retained) < 3:
        return ()

    subdiv = cv2.Subdiv2D((0, 0, width, height))
    for _, point in retained:
        try:
            subdiv.insert((float(point[0]), float(point[1])))
        except cv2.error:
            continue
    raw = subdiv.getTriangleList()
    if raw is None or len(raw) == 0:
        return ()

    retained_indices = np.asarray([item[0] for item in retained], dtype=np.int32)
    retained_points = np.asarray([item[1] for item in retained], dtype=np.float32)
    triangles: set[tuple[int, int, int]] = set()
    for triangle in np.asarray(raw, dtype=np.float32).reshape(-1, 3, 2):
        if not all(0.0 <= vertex[0] < width and 0.0 <= vertex[1] < height for vertex in triangle):
            continue
        mapped: list[int] = []
        for vertex in triangle:
            distances = np.sum((retained_points - vertex) ** 2, axis=1)
            nearest = int(np.argmin(distances))
            if distances[nearest] > 1.0:
                mapped = []
                break
            mapped.append(int(retained_indices[nearest]))
        if len(set(mapped)) == 3:
            triangles.add(tuple(sorted(mapped)))
    return tuple(sorted(triangles))


class GeometricFaceRenderer:
    """Warp continuously selected references onto a tracked facial landmark mesh."""

    def __init__(self, *, feather_fraction: float = 0.055) -> None:
        if not isfinite(feather_fraction) or not 0.0 < feather_fraction <= 0.25:
            raise ValueError("Feather fraction must be in (0, 0.25]")
        self._feather_fraction = feather_fraction
        self._format: FrameFormat | None = None
        self._triangles: tuple[tuple[int, int, int], ...] | None = None

    def open(self, output_format: FrameFormat) -> None:
        self._format = output_format
        self._triangles = None

    def render(
        self,
        face: FaceState,
        references: tuple[PreparedReference, ...],
        weights: tuple[ReferenceWeight, ...],
    ) -> RenderedFace:
        frame_format = self._format
        if frame_format is None:
            raise StageError("Geometric face renderer is not open")
        if face.landmark_schema != _LANDMARK_SCHEMA:
            raise StageError(f"Unsupported live landmark schema: {face.landmark_schema}")
        if len(face.landmarks) < 478:
            raise StageError("Geometric rendering requires the complete 478-point face mesh")
        if not references or not weights:
            raise StageError("Geometric rendering requires at least one selected reference")

        by_id = {reference.reference_id: reference for reference in references}
        if len(by_id) != len(references):
            raise StageError("Reference IDs must be unique during rendering")
        selected: list[tuple[PreparedReference, float]] = []
        total = 0.0
        for item in weights:
            reference = by_id.get(item.reference_id)
            if reference is None:
                raise StageError(f"Selected reference is unavailable: {item.reference_id}")
            if reference.landmark_schema != _LANDMARK_SCHEMA or len(reference.landmarks) < 478:
                raise StageError(f"Reference {item.reference_id} has an incompatible landmark mesh")
            if reference.rgb.ndim != 3 or reference.rgb.shape[2] != 3:
                raise StageError(f"Reference {item.reference_id} has an invalid RGB image")
            if item.weight > _WEIGHT_EPSILON:
                selected.append((reference, item.weight))
                total += item.weight
        if not selected or abs(total - 1.0) > 1e-4:
            raise StageError("Selected reference weights must be normalized")

        width, height = frame_format.width, frame_format.height
        source_points = {
            reference.reference_id: np.asarray(
                [(point.x, point.y) for point in reference.landmarks], dtype=np.float32
            )
            for reference, _ in selected
        }
        target_all = np.asarray([(point.x, point.y) for point in face.landmarks], np.float32)
        triangles = self._triangles
        if triangles is None:
            topology_reference = selected[0][0]
            reference_height, reference_width = topology_reference.rgb.shape[:2]
            mesh_source = source_points[topology_reference.reference_id][np.asarray(_MESH_INDICES)]
            local_triangles = delaunay_triangle_indices(
                mesh_source, reference_width, reference_height
            )
            triangles = tuple(
                tuple(_MESH_INDICES[local_index] for local_index in triangle)
                for triangle in local_triangles
            )
            if not triangles:
                raise StageError("Reference face does not contain a renderable landmark region")
            self._triangles = triangles
        canvas = np.zeros((height, width, 3), dtype=np.float32)
        coverage = np.zeros((height, width), dtype=np.uint8)
        for triangle in triangles:
            destination = target_all[np.asarray(triangle)].astype(np.float32)
            if abs(float(cv2.contourArea(destination))) < 0.5:
                continue
            x, y, triangle_width, triangle_height = cv2.boundingRect(destination)
            x0, y0 = max(0, x), max(0, y)
            x1, y1 = min(width, x + triangle_width), min(height, y + triangle_height)
            if x1 <= x0 or y1 <= y0:
                continue
            local_destination = destination - np.array((x0, y0), dtype=np.float32)
            triangle_mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
            cv2.fillConvexPoly(
                triangle_mask,
                np.rint(local_destination).astype(np.int32),
                255,
                lineType=cv2.LINE_AA,
            )
            blended = np.zeros((y1 - y0, x1 - x0, 3), dtype=np.float32)
            available_weight = 0.0
            for reference, weight in selected:
                source = source_points[reference.reference_id][np.asarray(triangle)]
                if abs(float(cv2.contourArea(source))) < 0.25:
                    continue
                transform = cv2.getAffineTransform(source, local_destination)
                warped = cv2.warpAffine(
                    reference.rgb,
                    transform,
                    (x1 - x0, y1 - y0),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REFLECT_101,
                )
                blended += warped.astype(np.float32) * weight
                available_weight += weight
            if available_weight <= _WEIGHT_EPSILON:
                continue
            blended /= available_weight
            mask = triangle_mask.astype(np.float32)[..., None] / 255.0
            region = canvas[y0:y1, x0:x1]
            region *= 1.0 - mask
            region += blended * mask
            coverage_region = coverage[y0:y1, x0:x1]
            np.maximum(coverage_region, triangle_mask, out=coverage_region)

        oval = target_all[np.asarray(_FACE_OVAL)]
        visible_oval = oval[
            (oval[:, 0] >= 0) & (oval[:, 0] < width) & (oval[:, 1] >= 0) & (oval[:, 1] < height)
        ]
        if len(visible_oval) < 3:
            raise StageError("Tracked facial region lies outside the output frame")
        hull = cv2.convexHull(visible_oval.astype(np.float32)).reshape(-1, 2)
        face_size = max(1.0, min(face.bounds.width, face.bounds.height))
        feather_width = max(1.0, face_size * self._feather_fraction)
        x0 = max(0, int(np.floor(np.min(hull[:, 0]))))
        y0 = max(0, int(np.floor(np.min(hull[:, 1]))))
        x1 = min(width, int(np.ceil(np.max(hull[:, 0]))) + 1)
        y1 = min(height, int(np.ceil(np.max(hull[:, 1]))) + 1)
        hard_mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
        local_hull = np.rint(hull - np.array((x0, y0), np.float32)).astype(np.int32)
        cv2.fillConvexPoly(hard_mask, local_hull, 255, lineType=cv2.LINE_8)
        distance = cv2.distanceTransform(hard_mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        local_alpha = np.clip(distance / feather_width, 0.0, 1.0).astype(np.float32)
        local_alpha *= (coverage[y0:y1, x0:x1] > 0).astype(np.float32)
        alpha = np.zeros((height, width), dtype=np.float32)
        alpha[y0:y1, x0:x1] = local_alpha

        rendered = np.clip(np.rint(canvas), 0, 255).astype(np.uint8)
        rendered = np.ascontiguousarray(rendered)
        alpha = np.ascontiguousarray(alpha)
        rendered.setflags(write=False)
        alpha.setflags(write=False)
        return RenderedFace(face.frame_id, face.timestamp_ns, rendered, alpha)

    def close(self) -> None:
        self._format = None
        self._triangles = None
