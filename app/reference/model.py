"""Extensible reference-slot and enrollment metadata records."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from math import isfinite
from types import MappingProxyType
from typing import Any

from app.pipeline.types import Point2D, Rect, RGBImage

MANIFEST_SCHEMA = "facelive-reference-library"
MANIFEST_VERSION = 1
NORMALIZED_FACE_SIZE = 512
LANDMARK_SCHEMA = "mediapipe-face-landmarker-478-v1"


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


@dataclass(frozen=True, slots=True)
class ReferenceSlot:
    slot_id: str
    label: str
    yaw: float
    pitch: float
    expression: str
    guidance: str
    required: bool = True

    def __post_init__(self) -> None:
        if not self.slot_id or not all(
            c.islower() or c.isdigit() or c in "-_" for c in self.slot_id
        ):
            raise ValueError("Reference slot IDs use lowercase letters, digits, '-' and '_'")
        if not self.label.strip() or not self.expression.strip() or not self.guidance.strip():
            raise ValueError("Reference slot label, expression, and guidance are required")
        if not isfinite(self.yaw) or not isfinite(self.pitch):
            raise ValueError("Reference pose values must be finite")
        if type(self.required) is not bool:
            raise ValueError("Reference slot required flag must be boolean")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.slot_id,
            "label": self.label,
            "yaw": self.yaw,
            "pitch": self.pitch,
            "expression": self.expression,
            "guidance": self.guidance,
            "required": self.required,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ReferenceSlot:
        try:
            return cls(
                slot_id=str(value["id"]),
                label=str(value["label"]),
                yaw=float(value["yaw"]),
                pitch=float(value["pitch"]),
                expression=str(value["expression"]),
                guidance=str(value["guidance"]),
                required=value.get("required", True),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid reference slot: {exc}") from exc


INITIAL_REFERENCE_SLOTS = (
    ReferenceSlot(
        "front-neutral",
        "Front neutral",
        0.0,
        0.0,
        "neutral",
        "Face forward; relax your expression.",
    ),
    ReferenceSlot(
        "front-smile", "Front smile", 0.0, 0.0, "smile", "Face forward with a natural smile."
    ),
    ReferenceSlot(
        "left-20", "Left 20°", 20.0, 0.0, "neutral", "Turn your head about 20° to your left."
    ),
    ReferenceSlot(
        "left-40", "Left 40°", 40.0, 0.0, "neutral", "Turn your head about 40° to your left."
    ),
    ReferenceSlot(
        "right-20", "Right 20°", -20.0, 0.0, "neutral", "Turn your head about 20° to your right."
    ),
    ReferenceSlot(
        "right-40", "Right 40°", -40.0, 0.0, "neutral", "Turn your head about 40° to your right."
    ),
    ReferenceSlot(
        "up", "Look slightly up", 0.0, -15.0, "neutral", "Keep facing forward and look slightly up."
    ),
    ReferenceSlot(
        "down",
        "Look slightly down",
        0.0,
        15.0,
        "neutral",
        "Keep facing forward and look slightly down.",
    ),
)


@dataclass(frozen=True, slots=True)
class ReferenceMetadata:
    slot_id: str
    source_name: str
    source_sha256: str
    original_width: int
    original_height: int
    face_bounds: Rect
    brightness: float
    sharpness: float
    face_coverage: float
    alignment_rotation_degrees: float
    created_at: str
    blendshapes: Mapping[str, float] = field(default_factory=dict)
    asset_path: str | None = None
    asset_sha256: str | None = None

    def __post_init__(self) -> None:
        if not self.slot_id or not self.source_name or not self.created_at:
            raise ValueError("Reference slot, source name, and creation time are required")
        if not _is_sha256(self.source_sha256):
            raise ValueError("Reference source SHA-256 is invalid")
        if self.original_width <= 0 or self.original_height <= 0:
            raise ValueError("Original reference dimensions must be positive")
        for name in ("brightness", "sharpness", "face_coverage", "alignment_rotation_degrees"):
            if not isfinite(getattr(self, name)):
                raise ValueError(f"Reference {name} must be finite")
        if self.face_bounds.width <= 0 or self.face_bounds.height <= 0:
            raise ValueError("Reference face bounds must be positive")
        if not all(
            isfinite(value)
            for value in (
                self.face_bounds.x,
                self.face_bounds.y,
                self.face_bounds.width,
                self.face_bounds.height,
            )
        ):
            raise ValueError("Reference face bounds must be finite")
        if (
            self.face_bounds.x < 0
            or self.face_bounds.y < 0
            or self.face_bounds.x + self.face_bounds.width > 1
            or self.face_bounds.y + self.face_bounds.height > 1
        ):
            raise ValueError("Reference face bounds must be inside the source image")
        if not 0 <= self.brightness <= 255 or self.sharpness < 0:
            raise ValueError("Reference quality measurements are out of range")
        if not 0 < self.face_coverage <= 1:
            raise ValueError("Reference face coverage is out of range")
        if not isinstance(self.blendshapes, Mapping):
            raise ValueError("Reference blendshapes must be a mapping")
        if not all(
            isfinite(float(value)) and 0 <= float(value) <= 1 for value in self.blendshapes.values()
        ):
            raise ValueError("Reference blendshapes must be finite values in [0, 1]")
        if (self.asset_path is None) != (self.asset_sha256 is None):
            raise ValueError("Reference asset path and SHA-256 must be present together")
        if self.asset_path is not None and not self.asset_path:
            raise ValueError("Reference asset path is empty")
        if self.asset_sha256 is not None and not _is_sha256(self.asset_sha256):
            raise ValueError("Reference asset SHA-256 is invalid")
        if not isinstance(self.blendshapes, MappingProxyType):
            object.__setattr__(self, "blendshapes", MappingProxyType(dict(self.blendshapes)))

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "slot_id": self.slot_id,
            "source_name": self.source_name,
            "source_sha256": self.source_sha256,
            "original_size": [self.original_width, self.original_height],
            "face_bounds": [
                self.face_bounds.x,
                self.face_bounds.y,
                self.face_bounds.width,
                self.face_bounds.height,
            ],
            "quality": {
                "brightness": self.brightness,
                "sharpness": self.sharpness,
                "face_coverage": self.face_coverage,
            },
            "alignment_rotation_degrees": self.alignment_rotation_degrees,
            "created_at": self.created_at,
            "blendshapes": dict(sorted(self.blendshapes.items())),
        }
        if self.asset_path is not None:
            value["asset"] = self.asset_path
        if self.asset_sha256 is not None:
            value["asset_sha256"] = self.asset_sha256
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ReferenceMetadata:
        try:
            size = value["original_size"]
            bounds = value["face_bounds"]
            quality = value["quality"]
            blendshapes = value.get("blendshapes", {})
            if not isinstance(blendshapes, Mapping):
                raise ValueError("blendshapes must be an object")
            return cls(
                slot_id=str(value["slot_id"]),
                source_name=str(value["source_name"]),
                source_sha256=str(value["source_sha256"]),
                original_width=int(size[0]),
                original_height=int(size[1]),
                face_bounds=Rect(*(float(item) for item in bounds)),
                brightness=float(quality["brightness"]),
                sharpness=float(quality["sharpness"]),
                face_coverage=float(quality["face_coverage"]),
                alignment_rotation_degrees=float(value["alignment_rotation_degrees"]),
                created_at=str(value["created_at"]),
                blendshapes={str(k): float(v) for k, v in blendshapes.items()},
                asset_path=str(value["asset"]) if "asset" in value else None,
                asset_sha256=str(value["asset_sha256"]) if "asset_sha256" in value else None,
            )
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid reference metadata: {exc}") from exc


@dataclass(frozen=True, slots=True)
class ProcessedReference:
    metadata: ReferenceMetadata
    rgb: RGBImage
    landmarks: tuple[Point2D, ...]
