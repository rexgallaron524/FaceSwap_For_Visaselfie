"""Continuous pose-space selection over prepared references."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from math import isfinite

from app.pipeline.types import FaceState, PreparedReference, ReferenceWeight

_AXIS_EPSILON = 1e-6
_WEIGHT_EPSILON = 1e-12

type PoseKey = tuple[float, float]


class PoseSpaceReferenceSelector:
    """Blend yaw, pitch, and expression without using roll or apparent scale."""

    def select(
        self, face: FaceState, references: tuple[PreparedReference, ...]
    ) -> tuple[ReferenceWeight, ...]:
        if not references:
            return ()
        self._validate_references(references)

        pose_groups: dict[PoseKey, list[PreparedReference]] = defaultdict(list)
        for reference in references:
            pose_groups[(reference.yaw, reference.pitch)].append(reference)

        pose_weights = self._pose_weights(face, tuple(pose_groups))
        reference_weights: dict[str, float] = defaultdict(float)
        for pose, pose_weight in pose_weights.items():
            for reference_id, expression_weight in self._expression_weights(
                face, pose_groups[pose]
            ).items():
                reference_weights[reference_id] += pose_weight * expression_weight

        return self._normalized(reference_weights, references)

    @staticmethod
    def _validate_references(references: tuple[PreparedReference, ...]) -> None:
        identifiers = [reference.reference_id for reference in references]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("Reference IDs must be unique during selection")
        for reference in references:
            if not isfinite(reference.yaw) or not isfinite(reference.pitch):
                raise ValueError("Reference pose coordinates must be finite")
            if not reference.expression.strip():
                raise ValueError("Reference expressions must be nonempty")

    def _pose_weights(self, face: FaceState, poses: tuple[PoseKey, ...]) -> dict[PoseKey, float]:
        yaw_axis = tuple(pose for pose in poses if abs(pose[1]) <= _AXIS_EPSILON)
        if yaw_axis:
            weights = self._linear_weights(face.pose.yaw, ((pose[0], pose) for pose in yaw_axis))
        else:
            nearest = self._nearest_pose(face, poses)
            weights = {nearest: 1.0}

        pitch = face.pose.pitch
        if abs(pitch) <= _AXIS_EPSILON:
            return weights
        pitch_axis = tuple(
            pose
            for pose in poses
            if abs(pose[0]) <= _AXIS_EPSILON and pose[1] * pitch > _AXIS_EPSILON
        )
        if not pitch_axis:
            return weights

        extreme = max(abs(pose[1]) for pose in pitch_axis)
        pitch_influence = min(1.0, abs(pitch) / extreme)
        pitch_weights = self._linear_weights(pitch, ((pose[1], pose) for pose in pitch_axis))
        combined = {pose: weight * (1.0 - pitch_influence) for pose, weight in weights.items()}
        for pose, weight in pitch_weights.items():
            combined[pose] = combined.get(pose, 0.0) + weight * pitch_influence
        return combined

    @staticmethod
    def _nearest_pose(face: FaceState, poses: tuple[PoseKey, ...]) -> PoseKey:
        return min(
            poses,
            key=lambda pose: (
                ((face.pose.yaw - pose[0]) / 20.0) ** 2 + ((face.pose.pitch - pose[1]) / 15.0) ** 2,
                pose,
            ),
        )

    @staticmethod
    def _linear_weights(
        value: float, anchors: Iterable[tuple[float, PoseKey]]
    ) -> dict[PoseKey, float]:
        ordered = sorted(anchors, key=lambda item: (item[0], item[1]))
        if not ordered:
            return {}
        if value <= ordered[0][0]:
            return {ordered[0][1]: 1.0}
        if value >= ordered[-1][0]:
            return {ordered[-1][1]: 1.0}
        for (lower_value, lower_pose), (upper_value, upper_pose) in zip(
            ordered, ordered[1:], strict=False
        ):
            if lower_value <= value <= upper_value:
                if upper_value == lower_value:
                    return {lower_pose: 1.0}
                upper_weight = (value - lower_value) / (upper_value - lower_value)
                return {lower_pose: 1.0 - upper_weight, upper_pose: upper_weight}
        raise RuntimeError("Could not bracket pose value")

    def _expression_weights(
        self, face: FaceState, references: list[PreparedReference]
    ) -> dict[str, float]:
        grouped: dict[str, list[PreparedReference]] = defaultdict(list)
        for reference in references:
            grouped[self._expression_name(reference.expression)].append(reference)

        active: dict[str, float] = {}
        for expression in grouped:
            if expression != "neutral":
                active[expression] = self._expression_activation(face, expression)

        neutral_strength = max(0.0, 1.0 - sum(active.values()))
        if "neutral" in grouped:
            active["neutral"] = neutral_strength
        total = sum(active.values())
        if total <= _WEIGHT_EPSILON:
            fallback = "neutral" if "neutral" in grouped else min(grouped)
            active = {fallback: 1.0}
            total = 1.0

        weights: dict[str, float] = {}
        for expression, expression_weight in active.items():
            variants = sorted(grouped[expression], key=lambda reference: reference.reference_id)
            per_reference = expression_weight / total / len(variants)
            for reference in variants:
                weights[reference.reference_id] = per_reference
        return weights

    @staticmethod
    def _expression_name(value: str) -> str:
        return value.strip().casefold().replace("-", "_").replace(" ", "_")

    @staticmethod
    def _expression_activation(face: FaceState, expression: str) -> float:
        if expression == "smile":
            values = [
                face.blendshapes[name]
                for name in ("mouth_smile_left", "mouth_smile_right")
                if name in face.blendshapes
            ]
            return sum(values) / len(values) if values else 0.0
        return face.blendshapes.get(expression, 0.0)

    @staticmethod
    def _normalized(
        raw: dict[str, float], references: tuple[PreparedReference, ...]
    ) -> tuple[ReferenceWeight, ...]:
        ordered = [
            (reference.reference_id, raw.get(reference.reference_id, 0.0))
            for reference in references
            if raw.get(reference.reference_id, 0.0) > _WEIGHT_EPSILON
        ]
        total = sum(weight for _, weight in ordered)
        if total <= _WEIGHT_EPSILON:
            return ()
        return tuple(
            ReferenceWeight(reference_id, weight / total) for reference_id, weight in ordered
        )
