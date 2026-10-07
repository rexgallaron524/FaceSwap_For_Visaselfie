import numpy as np
import pytest

from app.pipeline.types import (
    FaceState,
    HeadPose,
    NormalizedLandmark,
    Point2D,
    PreparedReference,
    Rect,
)
from app.reference.selector import PoseSpaceReferenceSelector


def reference(
    reference_id: str,
    yaw: float,
    pitch: float = 0.0,
    expression: str = "neutral",
) -> PreparedReference:
    rgb = np.zeros((2, 2, 3), dtype=np.uint8)
    rgb.setflags(write=False)
    return PreparedReference(
        reference_id,
        yaw,
        pitch,
        expression,
        rgb,
        (Point2D(1.0, 1.0),),
        "test-landmarks-v1",
    )


REFERENCES = (
    reference("front-neutral", 0.0),
    reference("front-smile", 0.0, expression="smile"),
    reference("left-20", 20.0),
    reference("left-40", 40.0),
    reference("right-20", -20.0),
    reference("right-40", -40.0),
    reference("up", 0.0, -15.0),
    reference("down", 0.0, 15.0),
)


def face(
    yaw: float,
    pitch: float = 0.0,
    *,
    roll: float = 0.0,
    scale: float = 0.25,
    smile: float = 0.0,
) -> FaceState:
    return FaceState(
        frame_id=1,
        timestamp_ns=2,
        bounds=Rect(10, 10, 100, 100),
        center=Point2D(60, 60),
        scale=scale,
        pose=HeadPose(yaw, pitch, roll),
        landmarks=(Point2D(60, 60),),
        normalized_landmarks=(NormalizedLandmark(0.5, 0.5),),
        landmark_schema="test-landmarks-v1",
        tracking_confidence=1.0,
        blendshapes={"mouth_smile_left": smile, "mouth_smile_right": smile},
    )


def selected(yaw: float, pitch: float = 0.0, **kwargs: float) -> dict[str, float]:
    weights = PoseSpaceReferenceSelector().select(face(yaw, pitch, **kwargs), REFERENCES)
    return {weight.reference_id: weight.weight for weight in weights}


def test_front_and_outer_yaw_boundaries_clamp_to_nearest_reference():
    assert selected(0.0) == {"front-neutral": 1.0}
    assert selected(-100.0) == {"right-40": 1.0}
    assert selected(100.0) == {"left-40": 1.0}


@pytest.mark.parametrize(
    ("yaw", "expected"),
    [
        (-10.0, {"front-neutral": 0.5, "right-20": 0.5}),
        (-30.0, {"right-20": 0.5, "right-40": 0.5}),
        (10.0, {"front-neutral": 0.5, "left-20": 0.5}),
        (30.0, {"left-20": 0.5, "left-40": 0.5}),
    ],
)
def test_yaw_interpolates_continuously_between_adjacent_anchors(yaw, expected):
    assert selected(yaw) == pytest.approx(expected)


def test_pitch_blends_up_and_down_with_the_current_yaw_selection():
    assert selected(0.0, -7.5) == pytest.approx({"front-neutral": 0.5, "up": 0.5})
    assert selected(0.0, 7.5) == pytest.approx({"front-neutral": 0.5, "down": 0.5})
    assert selected(-10.0, 7.5) == pytest.approx(
        {"front-neutral": 0.25, "right-20": 0.25, "down": 0.5}
    )


def test_smile_splits_only_pose_weight_that_has_a_smile_variant():
    assert selected(0.0, smile=0.75) == pytest.approx({"front-neutral": 0.25, "front-smile": 0.75})
    assert selected(-10.0, smile=0.5) == pytest.approx(
        {"front-neutral": 0.25, "front-smile": 0.25, "right-20": 0.5}
    )


def test_roll_and_scale_do_not_change_reference_selection():
    baseline = selected(-10.0, 4.0, roll=0.0, scale=0.2, smile=0.3)
    transformed = selected(-10.0, 4.0, roll=37.0, scale=0.8, smile=0.3)
    assert transformed == pytest.approx(baseline)


def test_missing_intermediate_reference_uses_remaining_pose_anchors_and_normalizes():
    references = tuple(item for item in REFERENCES if item.reference_id != "right-20")
    weights = PoseSpaceReferenceSelector().select(face(-20.0, 5.0, smile=0.4), references)
    values = {weight.reference_id: weight.weight for weight in weights}

    assert values == pytest.approx(
        {
            "front-neutral": 0.2,
            "front-smile": 2 / 15,
            "right-40": 1 / 3,
            "down": 1 / 3,
        }
    )
    assert sum(values.values()) == pytest.approx(1.0)
    assert all(value >= 0 for value in values.values())


def test_empty_library_returns_no_selection_and_duplicate_ids_are_rejected():
    selector = PoseSpaceReferenceSelector()
    assert selector.select(face(0.0), ()) == ()
    with pytest.raises(ValueError, match="unique"):
        selector.select(face(0.0), (REFERENCES[0], REFERENCES[0]))
