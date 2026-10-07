from math import cos, radians, sin

import numpy as np
import pytest

from app.pipeline.types import NormalizedLandmark
from app.tracking.geometry import (
    canonical_blendshape_name,
    derived_tracking_confidence,
    landmark_bounds,
    normalized_to_pixel,
    rotation_matrix_to_head_pose,
)


def test_normalized_landmarks_convert_to_full_frame_pixels_and_bounds():
    points = (
        NormalizedLandmark(0.25, 0.20, -0.1),
        NormalizedLandmark(0.75, 0.80, 0.1),
    )

    assert normalized_to_pixel(points[0], 1280, 720).x == pytest.approx(320)
    assert normalized_to_pixel(points[0], 1280, 720).y == pytest.approx(144)
    bounds = landmark_bounds(points, 1280, 720)
    assert (bounds.x, bounds.y, bounds.width, bounds.height) == pytest.approx((320, 144, 640, 432))


def test_rotation_matrix_converts_to_documented_pose_convention():
    yaw, pitch, roll = map(radians, (24.0, -13.0, 7.0))
    rx = np.array([[1, 0, 0], [0, cos(pitch), -sin(pitch)], [0, sin(pitch), cos(pitch)]])
    ry = np.array([[cos(yaw), 0, sin(yaw)], [0, 1, 0], [-sin(yaw), 0, cos(yaw)]])
    rz = np.array([[cos(roll), -sin(roll), 0], [sin(roll), cos(roll), 0], [0, 0, 1]])
    transform = np.eye(4)
    transform[:3, :3] = rz @ ry @ rx

    pose = rotation_matrix_to_head_pose(transform)

    assert (pose.yaw, pose.pitch, pose.roll) == pytest.approx((24.0, -13.0, 7.0))


def test_tracking_confidence_penalizes_small_or_out_of_frame_faces():
    centered = tuple(
        NormalizedLandmark(0.4 + (index % 2) * 0.2, 0.4 + (index // 2) * 0.2) for index in range(4)
    )
    small = tuple(
        NormalizedLandmark(0.49 + (index % 2) * 0.02, 0.49 + (index // 2) * 0.02)
        for index in range(4)
    )
    clipped = (*centered[:2], NormalizedLandmark(-0.1, 0.5), NormalizedLandmark(1.1, 0.5))

    assert derived_tracking_confidence(centered) == pytest.approx(1.0)
    assert derived_tracking_confidence(small) < derived_tracking_confidence(centered)
    assert derived_tracking_confidence(clipped) < derived_tracking_confidence(centered)


def test_blendshape_names_are_backend_neutral_snake_case():
    assert canonical_blendshape_name("mouthSmileLeft") == "mouth_smile_left"
    assert canonical_blendshape_name("eyeBlinkRight") == "eye_blink_right"
