from dataclasses import replace
from math import cos, pi, sin, sqrt

import numpy as np
import pytest

from app.compositing import AlphaFaceCompositor
from app.pipeline.types import (
    FaceState,
    FrameFormat,
    HeadPose,
    NormalizedLandmark,
    Point2D,
    PreparedReference,
    Rect,
    ReferenceWeight,
    RenderedFace,
    StageError,
    VideoFrame,
)
from app.rendering import GeometricFaceRenderer
from app.rendering.geometric import (
    _FACE_OVAL,
    delaunay_triangle_indices,
    expression_adjusted_landmarks,
)

SCHEMA = "mediapipe-face-landmarker-478-v1"


def normalized_mesh() -> np.ndarray:
    values = np.empty((478, 2), dtype=np.float32)
    golden_angle = pi * (3.0 - sqrt(5.0))
    for index in range(478):
        radius = 0.34 * sqrt((index + 0.5) / 478)
        angle = index * golden_angle
        values[index] = (0.5 + radius * cos(angle), 0.5 + radius * sin(angle) * 1.12)
    for position, index in enumerate(_FACE_OVAL):
        angle = -pi / 2 + 2 * pi * position / len(_FACE_OVAL)
        values[index] = (0.5 + 0.42 * cos(angle), 0.5 + 0.46 * sin(angle))
    return values


def reference(reference_id: str, color: tuple[int, int, int]) -> PreparedReference:
    size = 96
    mesh = normalized_mesh()
    rgb = np.empty((size, size, 3), dtype=np.uint8)
    rgb[:] = color
    rgb.setflags(write=False)
    return PreparedReference(
        reference_id,
        0.0,
        0.0,
        "neutral",
        rgb,
        tuple(Point2D(float(x * size), float(y * size)) for x, y in mesh),
        SCHEMA,
    )


def tracked_face() -> FaceState:
    mesh = normalized_mesh()
    angle = np.deg2rad(13.0)
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]],
        dtype=np.float32,
    )
    pixels = (mesh - 0.5) @ rotation.T * np.array((105.0, 90.0)) + np.array((83.0, 59.0))
    landmarks = tuple(Point2D(float(x), float(y)) for x, y in pixels)
    normalized = tuple(NormalizedLandmark(float(x / 160), float(y / 120)) for x, y in pixels)
    return FaceState(
        7,
        11,
        Rect(38.0, 16.0, 90.0, 86.0),
        Point2D(83.0, 59.0),
        90 / 160,
        HeadPose(12.0, -4.0, 13.0),
        landmarks,
        normalized,
        SCHEMA,
        1.0,
    )


def test_delaunay_indices_are_deterministic_and_ignore_out_of_bounds_points():
    points = np.array(((1, 1), (8, 1), (8, 8), (1, 8), (4.5, 4.5), (-2, 3)), np.float32)

    first = delaunay_triangle_indices(points, 10, 10)
    second = delaunay_triangle_indices(points, 10, 10)

    assert first == second
    assert len(first) == 4
    assert all(5 not in triangle for triangle in first)


def test_renderer_interpolates_references_and_produces_a_feathered_face_mask():
    renderer = GeometricFaceRenderer()
    renderer.open(FrameFormat(160, 120, 30))
    references = (
        reference("warm", (220, 30, 30)),
        reference("green", (30, 220, 30)),
    )

    rendered = renderer.render(
        tracked_face(),
        references,
        (ReferenceWeight("warm", 0.25), ReferenceWeight("green", 0.75)),
    )

    assert rendered.active_region is not None
    x, y, width, height = rendered.active_region
    assert rendered.rgb.shape == (height, width, 3)
    assert rendered.alpha.shape == (height, width)
    assert rendered.rgb.dtype == np.uint8
    assert rendered.alpha.dtype == np.float32
    assert rendered.alpha[59 - y, 83 - x] > 0.95
    assert not np.any((rendered.alpha > 0.0) & np.all(rendered.rgb == 0, axis=2))
    assert rendered.rgb[59 - y, 83 - x] == pytest.approx((78, 172, 30), abs=2)
    assert not rendered.rgb.flags.writeable
    assert not rendered.alpha.flags.writeable


def test_compositor_preserves_pixels_outside_alpha_and_frame_identity():
    original_pixels = np.full((12, 16, 3), (10, 20, 180), dtype=np.uint8)
    original_pixels.setflags(write=False)
    rendered_pixels = np.full((12, 16, 3), (220, 40, 20), dtype=np.uint8)
    rendered_pixels.setflags(write=False)
    alpha = np.zeros((12, 16), dtype=np.float32)
    alpha[3:9, 5:11] = 1.0
    alpha.setflags(write=False)
    original = VideoFrame(4, 99, original_pixels)
    rendered = RenderedFace(4, 99, rendered_pixels, alpha)

    output = AlphaFaceCompositor(color_match_strength=0.0).composite(original, rendered)

    assert (output.frame_id, output.timestamp_ns) == (4, 99)
    assert output.rgb[0, 0].tolist() == [10, 20, 180]
    assert output.rgb[6, 8].tolist() == [220, 40, 20]
    assert original.rgb[6, 8].tolist() == [10, 20, 180]
    assert not output.rgb.flags.writeable


def test_compositor_accepts_a_tightly_cropped_active_region():
    original_pixels = np.full((12, 16, 3), (10, 20, 180), dtype=np.uint8)
    original_pixels.setflags(write=False)
    rendered_pixels = np.full((6, 6, 3), (220, 40, 20), dtype=np.uint8)
    rendered_pixels.setflags(write=False)
    alpha = np.ones((6, 6), dtype=np.float32)
    alpha.setflags(write=False)

    output = AlphaFaceCompositor(color_match_strength=0.0).composite(
        VideoFrame(4, 99, original_pixels),
        RenderedFace(4, 99, rendered_pixels, alpha, (5, 3, 6, 6)),
    )

    assert output.rgb[0, 0].tolist() == [10, 20, 180]
    assert output.rgb[6, 8].tolist() == [220, 40, 20]


def test_expression_geometry_closes_eyes_opens_mouth_and_lifts_smile_corners():
    original_face = tracked_face()
    expressive = replace(
        original_face,
        blendshapes={
            "eye_blink_left": 1.0,
            "eye_blink_right": 1.0,
            "jaw_open": 1.0,
            "mouth_smile_left": 1.0,
            "mouth_smile_right": 1.0,
        },
    )
    points = np.asarray([(point.x, point.y) for point in expressive.landmarks], np.float32)

    adjusted = expression_adjusted_landmarks(expressive, points)

    assert abs(adjusted[159, 1] - adjusted[145, 1]) < abs(points[159, 1] - points[145, 1])
    assert abs(adjusted[386, 1] - adjusted[374, 1]) < abs(points[386, 1] - points[374, 1])
    assert adjusted[14, 1] - adjusted[13, 1] > points[14, 1] - points[13, 1]
    assert adjusted[61, 0] < points[61, 0]
    assert adjusted[291, 0] > points[291, 0]


def test_renderer_and_compositor_reject_incompatible_inputs():
    renderer = GeometricFaceRenderer()
    with pytest.raises(StageError, match="not open"):
        renderer.render(
            tracked_face(),
            (reference("one", (1, 2, 3)),),
            (ReferenceWeight("one", 1.0),),
        )

    pixels = np.zeros((4, 4, 3), dtype=np.uint8)
    pixels.setflags(write=False)
    alpha = np.ones((4, 4), dtype=np.float32)
    alpha.setflags(write=False)
    with pytest.raises(StageError, match="identities"):
        AlphaFaceCompositor().composite(VideoFrame(1, 2, pixels), RenderedFace(9, 2, pixels, alpha))
