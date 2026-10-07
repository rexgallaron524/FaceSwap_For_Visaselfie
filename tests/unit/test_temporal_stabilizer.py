import statistics

import pytest

from app.pipeline.types import (
    FaceState,
    HeadPose,
    NormalizedLandmark,
    Point2D,
    Rect,
    ReferenceWeight,
    StageError,
)
from app.stabilization import TemporalSmoothingConfig, TemporalStabilizer


def face(
    frame_id: int,
    timestamp_ns: int,
    *,
    x: float = 50.0,
    scale: float = 0.25,
    yaw: float = 0.0,
    confidence: float = 1.0,
    shapes: dict[str, float] | None = None,
) -> FaceState:
    return FaceState(
        frame_id,
        timestamp_ns,
        Rect(x - 20.0, 30.0, 40.0 * scale / 0.25, 50.0 * scale / 0.25),
        Point2D(x, 55.0),
        scale,
        HeadPose(yaw, yaw / 2.0, -yaw / 3.0),
        (Point2D(x - 10.0, 45.0), Point2D(x + 10.0, 65.0)),
        (
            NormalizedLandmark((x - 10.0) / 200.0, 0.45),
            NormalizedLandmark((x + 10.0) / 200.0, 0.65),
        ),
        "test-mesh-v1",
        confidence,
        shapes or {},
    )


def test_geometry_pose_landmarks_and_scale_are_smoothed_with_identity_preserved():
    stabilizer = TemporalStabilizer()
    first = stabilizer.update(face(0, 1_000_000_000))
    current = face(1, 1_033_333_333, x=80.0, scale=0.40, yaw=30.0)

    result = stabilizer.update(current)

    assert first.center.x == 50.0
    assert (result.frame_id, result.timestamp_ns) == (1, 1_033_333_333)
    assert 50.0 < result.center.x < 80.0
    assert 0.25 < result.scale < 0.40
    assert 0.0 < result.pose.yaw < 30.0
    assert first.landmarks[0].x < result.landmarks[0].x < current.landmarks[0].x


def test_low_confidence_measurement_moves_less_than_high_confidence_measurement():
    high = TemporalStabilizer()
    low = TemporalStabilizer()
    initial = face(0, 1_000_000_000)
    high.update(initial)
    low.update(initial)

    high_result = high.update(face(1, 1_033_333_333, x=80.0, confidence=1.0))
    low_result = low.update(face(1, 1_033_333_333, x=80.0, confidence=0.2))

    assert 50.0 < low_result.center.x < high_result.center.x < 80.0


def test_small_jitter_is_reduced_while_large_motion_remains_responsive():
    stabilizer = TemporalStabilizer()
    timestamp = 1_000_000_000
    stabilizer.update(face(0, timestamp, x=50.0))
    raw_positions: list[float] = []
    stable_positions: list[float] = []
    for frame_id in range(1, 21):
        timestamp += 33_333_333
        raw_x = 49.0 if frame_id % 2 else 51.0
        raw_positions.append(raw_x)
        stable_positions.append(stabilizer.update(face(frame_id, timestamp, x=raw_x)).center.x)

    assert statistics.pstdev(stable_positions[5:]) < statistics.pstdev(raw_positions[5:])
    timestamp += 33_333_333
    moved = stabilizer.update(face(21, timestamp, x=90.0))
    assert moved.center.x > 70.0


def test_blink_and_mouth_attack_remain_faster_than_smile_transition():
    stabilizer = TemporalStabilizer()
    names = {
        "eye_blink_left": 0.0,
        "eye_blink_right": 0.0,
        "jaw_open": 0.0,
        "mouth_smile_left": 0.0,
        "mouth_smile_right": 0.0,
    }
    stabilizer.update(face(0, 1_000_000_000, shapes=names))
    active = {name: 1.0 for name in names}

    result = stabilizer.update(face(1, 1_033_333_333, shapes=active))

    assert result.blendshapes["eye_blink_left"] > result.blendshapes["jaw_open"]
    assert result.blendshapes["jaw_open"] > result.blendshapes["mouth_smile_left"]


def test_reference_weights_transition_continuously_and_stay_normalized():
    stabilizer = TemporalStabilizer()
    assert stabilizer.smooth_weights((ReferenceWeight("front", 1.0),), 1_000_000_000) == (
        ReferenceWeight("front", 1.0),
    )

    result = stabilizer.smooth_weights((ReferenceWeight("right", 1.0),), 1_033_333_333)
    values = {item.reference_id: item.weight for item in result}

    assert 0.0 < values["right"] < 1.0
    assert 0.0 < values["front"] < 1.0
    assert sum(values.values()) == pytest.approx(1.0)


def test_short_tracking_gap_holds_geometry_then_expires_and_decays_mouth_activity():
    stabilizer = TemporalStabilizer()
    observed = stabilizer.update(
        face(0, 1_000_000_000, shapes={"jaw_open": 0.8, "mouth_smile_left": 0.6})
    )

    held = stabilizer.coast(1, 1_100_000_000)

    assert held is not None
    assert (held.frame_id, held.timestamp_ns) == (1, 1_100_000_000)
    assert held.center == observed.center
    assert 0.0 < held.tracking_confidence < observed.tracking_confidence
    assert 0.0 < held.blendshapes["jaw_open"] < 0.8
    assert held.blendshapes["mouth_smile_left"] < 0.6
    assert stabilizer.coast(2, 1_200_000_000) is None


def test_reset_and_invalid_inputs_are_explicit():
    with pytest.raises(ValueError, match="hold_seconds"):
        TemporalSmoothingConfig(hold_seconds=1.0, reset_gap_seconds=0.5)

    stabilizer = TemporalStabilizer()
    stabilizer.update(face(1, 1_000_000_000))
    with pytest.raises(StageError, match="timestamps"):
        stabilizer.update(face(2, 1_000_000_000))
    stabilizer.reset()
    assert stabilizer.coast(3, 1_100_000_000) is None
