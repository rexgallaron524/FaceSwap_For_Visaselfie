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
# A coarse deformation mesh keeps the semantic contours that materially move during pose,
# blink, and mouth animation. The previous every-sixth-landmark fill produced roughly 300
# tiny triangles and spent most frame time rebuilding equivalent affine maps.
_MESH_SUPPORT = (
    205,
    425,
)
_MESH_FEATURES = (
    # Eyes and brows.
    33,
    159,
    133,
    145,
    362,
    386,
    263,
    374,
    70,
    107,
    336,
    300,
    # Nose bridge and nostrils.
    1,
    4,
    168,
    327,
    # Outer and inner lips.
    61,
    0,
    17,
    291,
    308,
    14,
    87,
    78,
    13,
)
_MESH_INDICES = tuple(sorted(set(_FACE_OVAL[::4]) | set(_MESH_FEATURES) | set(_MESH_SUPPORT)))
_WEIGHT_EPSILON = 1e-8
_EYE_GROUPS = (
    ((33, 133), (7, 144, 145, 153, 154, 155, 157, 158, 159, 160, 161, 163, 173, 246)),
    ((362, 263), (249, 373, 374, 380, 381, 382, 384, 385, 386, 387, 388, 390, 398, 466)),
)
_MOUTH_POINTS = (
    0,
    13,
    14,
    17,
    37,
    39,
    40,
    61,
    78,
    80,
    81,
    82,
    84,
    87,
    88,
    91,
    95,
    146,
    178,
    181,
    185,
    191,
    267,
    269,
    270,
    291,
    308,
    310,
    311,
    312,
    314,
    317,
    318,
    321,
    324,
    375,
    402,
    405,
    409,
    415,
)
_INNER_LIP = (
    78,
    95,
    88,
    178,
    87,
    14,
    317,
    402,
    318,
    324,
    308,
    415,
    310,
    311,
    312,
    13,
    82,
    81,
    80,
    191,
)


def expression_adjusted_landmarks(face: FaceState, points: np.ndarray) -> np.ndarray:
    """Apply small blendshape-driven corrections to live landmark geometry."""
    adjusted = np.asarray(points, dtype=np.float32).copy()
    if adjusted.shape != (len(face.landmarks), 2):
        raise ValueError("Expression geometry must match the FaceState landmark count")

    blink_values = (
        float(face.blendshapes.get("eye_blink_right", 0.0)),
        float(face.blendshapes.get("eye_blink_left", 0.0)),
    )
    for (corners, contour), blink in zip(_EYE_GROUPS, blink_values, strict=True):
        closure = min(1.0, max(0.0, blink)) * 0.82
        if closure <= 0.0:
            continue
        center_y = float(np.mean(adjusted[np.asarray(corners), 1]))
        indices = np.asarray(contour)
        adjusted[indices, 1] += (center_y - adjusted[indices, 1]) * closure

    jaw_open = min(1.0, max(0.0, float(face.blendshapes.get("jaw_open", 0.0))))
    if jaw_open > 0.0:
        center_y = float((adjusted[13, 1] + adjusted[14, 1]) / 2.0)
        indices = np.asarray(_MOUTH_POINTS)
        adjusted[indices, 1] = center_y + (adjusted[indices, 1] - center_y) * (
            1.0 + 0.28 * jaw_open
        )
        minimum_gap = face.bounds.height * 0.032 * jaw_open
        gap = float(adjusted[14, 1] - adjusted[13, 1])
        if gap < minimum_gap:
            correction = (minimum_gap - gap) / 2.0
            adjusted[13, 1] -= correction
            adjusted[14, 1] += correction

    smile_values = [
        float(face.blendshapes[name])
        for name in ("mouth_smile_left", "mouth_smile_right")
        if name in face.blendshapes
    ]
    smile = sum(smile_values) / len(smile_values) if smile_values else 0.0
    if smile > 0.0:
        horizontal = face.bounds.width * 0.018 * smile
        vertical = face.bounds.height * 0.010 * smile
        adjusted[61] += (-horizontal, -vertical)
        adjusted[291] += (horizontal, -vertical)
    return adjusted


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


def _smooth_landmark_map(
    source_points: np.ndarray,
    destination_points: np.ndarray,
    width: int,
    height: int,
    origin: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a smooth destination-to-source map from sparse landmark displacement."""
    control_indices = np.asarray(_MESH_INDICES)
    source = np.asarray(source_points[control_indices], dtype=np.float32)
    destination = np.asarray(destination_points[control_indices], dtype=np.float32) - origin
    if source.shape != destination.shape or len(source) < 3:
        raise StageError("Geometric landmark map requires matching control points")

    destination_augmented = np.column_stack(
        (destination, np.ones(len(destination), dtype=np.float32))
    )
    affine, *_ = np.linalg.lstsq(destination_augmented, source, rcond=None)
    residual = source - destination_augmented @ affine

    longest = max(width, height)
    coarse_width = max(2, min(width, round(width / longest * 32)))
    coarse_height = max(2, min(height, round(height / longest * 32)))
    grid_x, grid_y = np.meshgrid(
        np.linspace(0.0, width - 1.0, coarse_width, dtype=np.float32),
        np.linspace(0.0, height - 1.0, coarse_height, dtype=np.float32),
    )
    query = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    delta = query[:, None, :] - destination[None, :, :]
    distance_squared = np.sum(delta * delta, axis=2)
    weights = 1.0 / np.maximum(distance_squared, 4.0)
    weights *= weights
    local_residual = weights @ residual / np.sum(weights, axis=1, keepdims=True)
    query_augmented = np.column_stack((query, np.ones(len(query), dtype=np.float32)))
    mapped = query_augmented @ affine + local_residual
    coarse_x = mapped[:, 0].reshape(coarse_height, coarse_width).astype(np.float32)
    coarse_y = mapped[:, 1].reshape(coarse_height, coarse_width).astype(np.float32)
    map_x = cv2.resize(coarse_x, (width, height), interpolation=cv2.INTER_CUBIC)
    map_y = cv2.resize(coarse_y, (width, height), interpolation=cv2.INTER_CUBIC)
    return map_x, map_y


class GeometricFaceRenderer:
    """Warp continuously selected references onto a tracked facial landmark mesh."""

    def __init__(self, *, feather_fraction: float = 0.055, working_resolution: int = 192) -> None:
        if not isfinite(feather_fraction) or not 0.0 < feather_fraction <= 0.25:
            raise ValueError("Feather fraction must be in (0, 0.25]")
        if type(working_resolution) is not int or working_resolution < 128:
            raise ValueError("Working resolution must be an integer of at least 128 pixels")
        self._feather_fraction = feather_fraction
        self._working_resolution = working_resolution
        self._format: FrameFormat | None = None
        self._reference_signature: tuple[tuple[object, ...], ...] = ()
        self._canonical_points: np.ndarray | None = None
        self._canonical_images: dict[str, np.ndarray] = {}

    def open(self, output_format: FrameFormat) -> None:
        self._format = output_format
        self._reference_signature = ()
        self._canonical_points = None
        self._canonical_images.clear()

    @staticmethod
    def _signature(references: tuple[PreparedReference, ...]) -> tuple[tuple[object, ...], ...]:
        return tuple(
            (
                reference.reference_id,
                id(reference.rgb),
                id(reference.landmarks),
                reference.rgb.shape,
                reference.landmark_schema,
            )
            for reference in references
        )

    def _prepare_reference_cache(
        self,
        references: tuple[PreparedReference, ...],
        source_points: dict[str, np.ndarray],
    ) -> None:
        signature = self._signature(references)
        if signature == self._reference_signature:
            return
        canonical = references[0]
        source_height, source_width = canonical.rgb.shape[:2]
        scale = min(1.0, self._working_resolution / max(source_width, source_height))
        canonical_width = max(1, round(source_width * scale))
        canonical_height = max(1, round(source_height * scale))
        scale_xy = np.array(
            (canonical_width / source_width, canonical_height / source_height), dtype=np.float32
        )
        working_points = {
            reference.reference_id: points * scale_xy
            for reference, points in (
                (reference, source_points[reference.reference_id]) for reference in references
            )
        }
        canonical_points = working_points[canonical.reference_id]
        zero_origin = np.zeros(2, dtype=np.float32)
        images: dict[str, np.ndarray] = {}
        for reference in references:
            if reference.rgb.shape[:2] != (source_height, source_width):
                raise StageError("Geometric references must share one normalized image size")
            resized = cv2.resize(
                reference.rgb,
                (canonical_width, canonical_height),
                interpolation=cv2.INTER_AREA,
            )
            if reference.reference_id == canonical.reference_id:
                aligned = resized
            else:
                map_x, map_y = _smooth_landmark_map(
                    working_points[reference.reference_id],
                    canonical_points,
                    canonical_width,
                    canonical_height,
                    zero_origin,
                )
                aligned = cv2.remap(
                    resized,
                    map_x,
                    map_y,
                    interpolation=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REFLECT_101,
                )
            images[reference.reference_id] = aligned.astype(np.float32)
        self._reference_signature = signature
        self._canonical_points = canonical_points
        self._canonical_images = images

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
            for reference in references
        }
        target_all = expression_adjusted_landmarks(
            face, np.asarray([(point.x, point.y) for point in face.landmarks], np.float32)
        )
        self._prepare_reference_cache(references, source_points)
        canonical_points = self._canonical_points
        if canonical_points is None:
            raise StageError("Reference appearance cache is unavailable")
        blended_reference = np.zeros_like(next(iter(self._canonical_images.values())))
        for reference, weight in selected:
            blended_reference += self._canonical_images[reference.reference_id] * weight

        oval = target_all[np.asarray(_FACE_OVAL)]
        visible_oval = oval[
            (oval[:, 0] >= 0) & (oval[:, 0] < width) & (oval[:, 1] >= 0) & (oval[:, 1] < height)
        ]
        if len(visible_oval) < 3:
            raise StageError("Tracked facial region lies outside the output frame")
        hull = cv2.convexHull(visible_oval.astype(np.float32)).reshape(-1, 2)
        x0 = max(0, int(np.floor(np.min(hull[:, 0]))))
        y0 = max(0, int(np.floor(np.min(hull[:, 1]))))
        x1 = min(width, int(np.ceil(np.max(hull[:, 0]))) + 1)
        y1 = min(height, int(np.ceil(np.max(hull[:, 1]))) + 1)
        roi_width, roi_height = x1 - x0, y1 - y0
        if roi_width <= 0 or roi_height <= 0:
            raise StageError("Tracked facial region lies outside the output frame")

        roi_origin = np.array((x0, y0), dtype=np.float32)
        render_scale = min(1.0, self._working_resolution / max(roi_width, roi_height))
        render_width = max(1, round(roi_width * render_scale))
        render_height = max(1, round(roi_height * render_scale))
        render_points = (target_all - roi_origin) * render_scale
        map_x, map_y = _smooth_landmark_map(
            canonical_points,
            render_points,
            render_width,
            render_height,
            np.zeros(2, dtype=np.float32),
        )
        canvas = cv2.remap(
            blended_reference,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )

        jaw_open = min(1.0, max(0.0, float(face.blendshapes.get("jaw_open", 0.0))))
        if jaw_open > 0.08:
            mouth = np.rint(render_points[np.asarray(_INNER_LIP)]).astype(np.int32)
            mouth_mask = np.zeros((render_height, render_width), dtype=np.uint8)
            cv2.fillConvexPoly(mouth_mask, cv2.convexHull(mouth), 255, lineType=cv2.LINE_AA)
            mouth_opacity = mouth_mask.astype(np.float32)[..., None] / 255.0 * (0.58 * jaw_open)
            cavity_color = np.array((38.0, 12.0, 18.0), dtype=np.float32)
            canvas = canvas * (1.0 - mouth_opacity) + cavity_color * mouth_opacity

        face_size = max(1.0, min(face.bounds.width, face.bounds.height))
        feather_width = max(1.0, face_size * self._feather_fraction * render_scale)
        hard_mask = np.zeros((render_height, render_width), dtype=np.uint8)
        local_hull = np.rint((hull - roi_origin) * render_scale).astype(np.int32)
        cv2.fillConvexPoly(hard_mask, local_hull, 255, lineType=cv2.LINE_8)
        distance = cv2.distanceTransform(hard_mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        local_alpha = np.clip(distance / feather_width, 0.0, 1.0).astype(np.float32)
        local_rendered = np.clip(np.rint(canvas), 0, 255).astype(np.uint8)
        if (render_width, render_height) != (roi_width, roi_height):
            local_rendered = cv2.resize(
                local_rendered, (roi_width, roi_height), interpolation=cv2.INTER_LINEAR
            )
            local_alpha = cv2.resize(
                local_alpha, (roi_width, roi_height), interpolation=cv2.INTER_LINEAR
            ).astype(np.float32, copy=False)

        local_rendered.setflags(write=False)
        local_alpha.setflags(write=False)
        return RenderedFace(
            face.frame_id,
            face.timestamp_ns,
            local_rendered,
            local_alpha,
            (x0, y0, roi_width, roi_height),
        )

    def close(self) -> None:
        self._format = None
        self._reference_signature = ()
        self._canonical_points = None
        self._canonical_images.clear()
