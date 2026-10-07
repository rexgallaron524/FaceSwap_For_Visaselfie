"""Time-based temporal smoothing for geometry, expressions, and reference weights."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, hypot, isfinite, log

from app.pipeline.types import (
    FaceState,
    HeadPose,
    NormalizedLandmark,
    Point2D,
    Rect,
    ReferenceWeight,
    StageError,
)

_RESPONSIVE_LANDMARKS = frozenset(
    (
        # Eye contours.
        7,
        33,
        133,
        144,
        145,
        153,
        154,
        155,
        157,
        158,
        159,
        160,
        161,
        163,
        173,
        246,
        249,
        263,
        362,
        373,
        374,
        380,
        381,
        382,
        384,
        385,
        386,
        387,
        388,
        390,
        398,
        466,
        # Inner and outer lips.
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
)


@dataclass(frozen=True, slots=True)
class TemporalSmoothingConfig:
    """Seconds-based constants tuned for a responsive 25–30 FPS preview."""

    translation_tau: float = 0.065
    scale_tau: float = 0.090
    pose_tau: float = 0.080
    landmark_tau: float = 0.055
    feature_landmark_tau: float = 0.032
    expression_tau: float = 0.075
    smile_attack_tau: float = 0.090
    smile_release_tau: float = 0.140
    blink_attack_tau: float = 0.018
    blink_release_tau: float = 0.045
    mouth_attack_tau: float = 0.035
    mouth_release_tau: float = 0.065
    reference_weight_tau: float = 0.110
    hold_seconds: float = 0.150
    reset_gap_seconds: float = 0.750

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if not isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.hold_seconds >= self.reset_gap_seconds:
            raise ValueError("hold_seconds must be shorter than reset_gap_seconds")


def _ema_alpha(delta_seconds: float, tau: float, motion_boost: float = 0.0) -> float:
    base = 1.0 - exp(-delta_seconds / tau)
    return min(1.0, base + (1.0 - base) * 0.45 * max(0.0, min(1.0, motion_boost)))


def _lerp(previous: float, current: float, alpha: float) -> float:
    return previous + (current - previous) * alpha


def _lerp_angle(previous: float, current: float, alpha: float) -> float:
    difference = (current - previous + 180.0) % 360.0 - 180.0
    return previous + difference * alpha


class TemporalStabilizer:
    """Smooth measurements without changing their coordinate or identity contracts."""

    def __init__(self, config: TemporalSmoothingConfig | None = None) -> None:
        self.config = config or TemporalSmoothingConfig()
        self._last_face: FaceState | None = None
        self._last_observed_ns: int | None = None
        self._weights: dict[str, float] = {}
        self._weight_timestamp_ns: int | None = None

    def update(self, face: FaceState) -> FaceState:
        previous = self._last_face
        if previous is None:
            self._accept_initial(face)
            return face
        if face.timestamp_ns <= previous.timestamp_ns:
            raise StageError("Stabilizer input timestamps must increase")
        delta_seconds = (face.timestamp_ns - previous.timestamp_ns) / 1_000_000_000
        if (
            delta_seconds >= self.config.reset_gap_seconds
            or face.landmark_schema != previous.landmark_schema
            or len(face.landmarks) != len(previous.landmarks)
        ):
            self._accept_initial(face)
            self._weights.clear()
            self._weight_timestamp_ns = None
            return face

        confidence_factor = 0.15 + 0.85 * face.tracking_confidence**2
        face_width = max(face.bounds.width, previous.bounds.width, 1.0)
        translation_speed = (
            hypot(face.center.x - previous.center.x, face.center.y - previous.center.y)
            / face_width
            / delta_seconds
        )
        translation_alpha = _ema_alpha(
            delta_seconds, self.config.translation_tau, translation_speed / 1.5
        )
        translation_alpha = max(0.02, translation_alpha * confidence_factor)

        scale_speed = abs(log(max(face.scale, 1e-8) / max(previous.scale, 1e-8))) / delta_seconds
        scale_alpha = _ema_alpha(delta_seconds, self.config.scale_tau, scale_speed / 1.0)
        scale_alpha = max(0.02, scale_alpha * confidence_factor)

        pose_speed = (
            max(
                abs(face.pose.yaw - previous.pose.yaw),
                abs(face.pose.pitch - previous.pose.pitch),
                abs(face.pose.roll - previous.pose.roll),
            )
            / delta_seconds
        )
        pose_alpha = _ema_alpha(delta_seconds, self.config.pose_tau, pose_speed / 180.0)
        pose_alpha = max(0.02, pose_alpha * confidence_factor)

        center = Point2D(
            _lerp(previous.center.x, face.center.x, translation_alpha),
            _lerp(previous.center.y, face.center.y, translation_alpha),
        )
        width = _lerp(previous.bounds.width, face.bounds.width, scale_alpha)
        height = _lerp(previous.bounds.height, face.bounds.height, scale_alpha)
        bounds = Rect(center.x - width / 2.0, center.y - height / 2.0, width, height)
        pose = HeadPose(
            _lerp_angle(previous.pose.yaw, face.pose.yaw, pose_alpha),
            _lerp_angle(previous.pose.pitch, face.pose.pitch, pose_alpha),
            _lerp_angle(previous.pose.roll, face.pose.roll, pose_alpha),
        )

        landmark_motion = (
            max(
                hypot(current.x - old.x, current.y - old.y)
                for old, current in zip(previous.landmarks, face.landmarks, strict=True)
            )
            / face_width
        )
        regular_alpha = _ema_alpha(
            delta_seconds,
            self.config.landmark_tau,
            landmark_motion / max(delta_seconds, 1e-6) / 2.0,
        )
        regular_alpha = max(0.03, regular_alpha * confidence_factor)
        feature_alpha = _ema_alpha(delta_seconds, self.config.feature_landmark_tau)
        feature_alpha = max(regular_alpha, feature_alpha * confidence_factor)

        def landmark_alpha(index: int) -> float:
            return feature_alpha if index in _RESPONSIVE_LANDMARKS else regular_alpha

        landmarks = tuple(
            Point2D(
                _lerp(old.x, current.x, landmark_alpha(index)),
                _lerp(old.y, current.y, landmark_alpha(index)),
            )
            for index, (old, current) in enumerate(
                zip(previous.landmarks, face.landmarks, strict=True)
            )
        )
        normalized_landmarks = tuple(
            NormalizedLandmark(
                _lerp(old.x, current.x, landmark_alpha(index)),
                _lerp(old.y, current.y, landmark_alpha(index)),
                _lerp(old.z, current.z, regular_alpha),
            )
            for index, (old, current) in enumerate(
                zip(previous.normalized_landmarks, face.normalized_landmarks, strict=True)
            )
        )
        blendshapes = {
            name: _lerp(
                previous.blendshapes.get(name, value),
                value,
                self._expression_alpha(
                    name,
                    value >= previous.blendshapes.get(name, value),
                    delta_seconds,
                    confidence_factor,
                ),
            )
            for name, value in face.blendshapes.items()
        }
        smoothed = FaceState(
            face.frame_id,
            face.timestamp_ns,
            bounds,
            center,
            _lerp(previous.scale, face.scale, scale_alpha),
            pose,
            landmarks,
            normalized_landmarks,
            face.landmark_schema,
            face.tracking_confidence,
            blendshapes,
        )
        self._last_face = smoothed
        self._last_observed_ns = face.timestamp_ns
        return smoothed

    def smooth_weights(
        self, weights: tuple[ReferenceWeight, ...], timestamp_ns: int
    ) -> tuple[ReferenceWeight, ...]:
        if not weights:
            self._weights.clear()
            self._weight_timestamp_ns = None
            return ()
        if self._weight_timestamp_ns is None or not self._weights:
            self._weights = {item.reference_id: item.weight for item in weights}
            self._weight_timestamp_ns = timestamp_ns
            return weights
        if timestamp_ns <= self._weight_timestamp_ns:
            raise StageError("Reference-weight timestamps must increase")
        delta_seconds = (timestamp_ns - self._weight_timestamp_ns) / 1_000_000_000
        alpha = _ema_alpha(delta_seconds, self.config.reference_weight_tau)
        target = {item.reference_id: item.weight for item in weights}
        ordered_ids = [item.reference_id for item in weights]
        ordered_ids.extend(
            sorted(identifier for identifier in self._weights if identifier not in target)
        )
        smoothed = {
            identifier: _lerp(
                self._weights.get(identifier, 0.0), target.get(identifier, 0.0), alpha
            )
            for identifier in ordered_ids
        }
        total = sum(value for value in smoothed.values() if value > 1e-8)
        if total <= 1e-8:
            self._weights.clear()
            self._weight_timestamp_ns = timestamp_ns
            return ()
        result = tuple(
            ReferenceWeight(identifier, value / total)
            for identifier, value in smoothed.items()
            if value > 1e-8
        )
        self._weights = {item.reference_id: item.weight for item in result}
        self._weight_timestamp_ns = timestamp_ns
        return result

    def coast(self, frame_id: int, timestamp_ns: int) -> FaceState | None:
        previous = self._last_face
        observed_ns = self._last_observed_ns
        if previous is None or observed_ns is None or timestamp_ns <= previous.timestamp_ns:
            return None
        elapsed = (timestamp_ns - observed_ns) / 1_000_000_000
        if elapsed > self.config.hold_seconds:
            return None
        retention = max(0.0, 1.0 - elapsed / self.config.hold_seconds)
        dynamic_prefixes = ("eye_blink_", "jaw_", "mouth_")
        blendshapes = {
            name: value * retention if name.startswith(dynamic_prefixes) else value
            for name, value in previous.blendshapes.items()
        }
        held = FaceState(
            frame_id,
            timestamp_ns,
            previous.bounds,
            previous.center,
            previous.scale,
            previous.pose,
            previous.landmarks,
            previous.normalized_landmarks,
            previous.landmark_schema,
            previous.tracking_confidence * retention,
            blendshapes,
        )
        self._last_face = held
        return held

    def reset(self) -> None:
        self._last_face = None
        self._last_observed_ns = None
        self._weights.clear()
        self._weight_timestamp_ns = None

    def _accept_initial(self, face: FaceState) -> None:
        self._last_face = face
        self._last_observed_ns = face.timestamp_ns

    def _expression_alpha(
        self, name: str, increasing: bool, delta_seconds: float, confidence_factor: float
    ) -> float:
        if name.startswith("eye_blink_"):
            tau = self.config.blink_attack_tau if increasing else self.config.blink_release_tau
        elif name in ("mouth_smile_left", "mouth_smile_right"):
            tau = self.config.smile_attack_tau if increasing else self.config.smile_release_tau
        elif name == "jaw_open" or name.startswith(("mouth_funnel", "mouth_pucker")):
            tau = self.config.mouth_attack_tau if increasing else self.config.mouth_release_tau
        else:
            tau = self.config.expression_tau
        return max(0.03, _ema_alpha(delta_seconds, tau) * confidence_factor)
