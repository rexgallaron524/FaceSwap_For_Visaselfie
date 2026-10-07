"""In-memory enrollment library with portable, checksum-validated persistence."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import replace
from datetime import UTC, datetime
from math import isfinite
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.pipeline.types import Point2D, PreparedReference, StageError
from app.reference.model import (
    INITIAL_REFERENCE_SLOTS,
    LANDMARK_SCHEMA,
    MANIFEST_SCHEMA,
    MANIFEST_VERSION,
    NORMALIZED_FACE_SIZE,
    ProcessedReference,
    ReferenceMetadata,
    ReferenceSlot,
)
from app.reference.preprocessor import ReferencePreprocessor


def serialize_manifest(value: Mapping[str, Any]) -> str:
    """Return the canonical UTF-8 JSON representation used on disk."""
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def parse_manifest(text: str) -> dict[str, Any]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise StageError(f"Reference manifest is not valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise StageError("Reference manifest root must be an object")
    return value


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


class ReferenceLibraryStore:
    def __init__(
        self,
        preprocessor: ReferencePreprocessor | None = None,
        slots: tuple[ReferenceSlot, ...] = INITIAL_REFERENCE_SLOTS,
        clock=_utc_now,
    ) -> None:
        self._preprocessor = preprocessor or ReferencePreprocessor()
        self._slots = self._validate_slots(slots)
        self._clock = clock
        self._entries: dict[str, ProcessedReference] = {}
        self._manifest_path: Path | None = None

    @staticmethod
    def _validate_slots(slots: tuple[ReferenceSlot, ...]) -> tuple[ReferenceSlot, ...]:
        if not slots:
            raise ValueError("A reference library needs at least one slot")
        identifiers = [slot.slot_id for slot in slots]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("Reference slot IDs must be unique")
        return tuple(slots)

    def slots(self) -> tuple[ReferenceSlot, ...]:
        return self._slots

    def slot(self, slot_id: str) -> ReferenceSlot:
        for slot in self._slots:
            if slot.slot_id == slot_id:
                return slot
        raise KeyError(slot_id)

    def add_slot(self, slot: ReferenceSlot) -> None:
        if any(existing.slot_id == slot.slot_id for existing in self._slots):
            raise ValueError(f"Reference slot already exists: {slot.slot_id}")
        self._slots = (*self._slots, slot)

    def add_reference(self, slot_id: str, image_path: Path) -> ReferenceMetadata:
        try:
            slot = self.slot(slot_id)
        except KeyError as exc:
            raise StageError(f"Unknown reference slot: {slot_id}") from exc
        processed = self._preprocessor.process(slot, image_path)
        self._entries[slot_id] = processed
        return processed.metadata

    def remove_reference(self, slot_id: str) -> None:
        self._entries.pop(slot_id, None)

    def metadata(self, slot_id: str) -> ReferenceMetadata | None:
        entry = self._entries.get(slot_id)
        return entry.metadata if entry is not None else None

    def processed(self, slot_id: str) -> ProcessedReference | None:
        return self._entries.get(slot_id)

    def reference(self, slot_id: str) -> PreparedReference | None:
        entry = self._entries.get(slot_id)
        if entry is None:
            return None
        return self._prepared(self.slot(slot_id), entry)

    def references(self) -> tuple[PreparedReference, ...]:
        return tuple(
            self._prepared(slot, self._entries[slot.slot_id])
            for slot in self._slots
            if slot.slot_id in self._entries
        )

    @staticmethod
    def _prepared(slot: ReferenceSlot, entry: ProcessedReference) -> PreparedReference:
        return PreparedReference(
            reference_id=slot.slot_id,
            yaw=slot.yaw,
            pitch=slot.pitch,
            expression=slot.expression,
            rgb=entry.rgb,
            landmarks=entry.landmarks,
            landmark_schema=LANDMARK_SCHEMA,
        )

    @property
    def valid_count(self) -> int:
        return len(self._entries)

    @property
    def required_count(self) -> int:
        return sum(slot.required for slot in self._slots)

    @property
    def completed_required_count(self) -> int:
        return sum(slot.required and slot.slot_id in self._entries for slot in self._slots)

    @property
    def is_complete(self) -> bool:
        return all(not slot.required or slot.slot_id in self._entries for slot in self._slots)

    @property
    def manifest_path(self) -> Path | None:
        return self._manifest_path

    def clear(self) -> None:
        self._entries.clear()
        self._manifest_path = None

    def save(self, path: Path) -> None:
        if not self._entries:
            raise StageError("Add at least one valid reference before saving")
        path = path.resolve()
        if path.suffix.casefold() != ".json":
            path = path.with_suffix(".json")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            asset_directory = path.parent / f"{path.stem}_assets"
            asset_directory.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise StageError(f"Could not create reference-library directory: {exc}") from exc

        references: list[dict[str, Any]] = []
        updated_entries: dict[str, ProcessedReference] = {}
        for slot in self._slots:
            entry = self._entries.get(slot.slot_id)
            asset_file = asset_directory / f"{slot.slot_id}.png"
            if entry is None:
                try:
                    asset_file.unlink(missing_ok=True)
                except OSError as exc:
                    raise StageError(
                        f"Could not remove cached reference {slot.label}: {exc}"
                    ) from exc
                continue
            ok, encoded = cv2.imencode(".png", cv2.cvtColor(entry.rgb, cv2.COLOR_RGB2BGR))
            if not ok:
                raise StageError(f"Could not encode normalized reference: {slot.label}")
            asset_bytes = encoded.tobytes()
            try:
                asset_file.write_bytes(asset_bytes)
            except OSError as exc:
                raise StageError(
                    f"Could not save normalized reference {slot.label}: {exc}"
                ) from exc
            relative_asset = asset_file.relative_to(path.parent).as_posix()
            metadata = replace(
                entry.metadata,
                asset_path=relative_asset,
                asset_sha256=hashlib.sha256(asset_bytes).hexdigest(),
            )
            updated = ProcessedReference(metadata, entry.rgb, entry.landmarks)
            updated_entries[slot.slot_id] = updated
            references.append(
                {
                    "metadata": metadata.to_dict(),
                    "landmarks": [[point.x, point.y] for point in entry.landmarks],
                }
            )

        manifest = {
            "schema": MANIFEST_SCHEMA,
            "version": MANIFEST_VERSION,
            "saved_at": self._clock(),
            "normalization": {
                "width": NORMALIZED_FACE_SIZE,
                "height": NORMALIZED_FACE_SIZE,
                "landmark_schema": LANDMARK_SCHEMA,
            },
            "slots": [slot.to_dict() for slot in self._slots],
            "references": references,
        }
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        try:
            temporary.write_text(serialize_manifest(manifest), encoding="utf-8", newline="\n")
            temporary.replace(path)
        except OSError as exc:
            temporary.unlink(missing_ok=True)
            raise StageError(f"Could not save reference manifest: {exc}") from exc
        self._entries = updated_entries
        self._manifest_path = path

    def load(self, path: Path) -> None:
        """Load cached normalized assets; failed loads always leave the library empty."""
        self.clear()
        path = path.resolve()
        try:
            if path.stat().st_size > 4 * 1024 * 1024:
                raise StageError("Reference manifest is unexpectedly large")
            manifest = parse_manifest(path.read_text(encoding="utf-8"))
            slots, entries = self._load_manifest(path, manifest)
        except StageError:
            raise
        except (OSError, ValueError, TypeError, KeyError) as exc:
            raise StageError(f"Could not load reference library: {exc}") from exc
        self._slots = slots
        self._entries = entries
        self._manifest_path = path

    def _load_manifest(
        self, path: Path, manifest: Mapping[str, Any]
    ) -> tuple[tuple[ReferenceSlot, ...], dict[str, ProcessedReference]]:
        if manifest.get("schema") != MANIFEST_SCHEMA:
            raise StageError("Unsupported reference-library schema")
        if manifest.get("version") != MANIFEST_VERSION:
            raise StageError(f"Unsupported reference-library version: {manifest.get('version')}")
        normalization = manifest.get("normalization")
        if not isinstance(normalization, Mapping):
            raise StageError("Reference manifest is missing normalization metadata")
        expected = (NORMALIZED_FACE_SIZE, NORMALIZED_FACE_SIZE, LANDMARK_SCHEMA)
        actual = (
            normalization.get("width"),
            normalization.get("height"),
            normalization.get("landmark_schema"),
        )
        if actual != expected:
            raise StageError("Reference normalization format is incompatible")

        raw_slots = manifest.get("slots")
        raw_references = manifest.get("references")
        if not isinstance(raw_slots, list) or not isinstance(raw_references, list):
            raise StageError("Reference manifest slots and references must be lists")
        try:
            slots = self._validate_slots(tuple(ReferenceSlot.from_dict(item) for item in raw_slots))
        except (TypeError, ValueError) as exc:
            raise StageError(str(exc)) from exc
        slot_ids = {slot.slot_id for slot in slots}
        entries: dict[str, ProcessedReference] = {}
        root = path.parent.resolve()
        for raw_reference in raw_references:
            if not isinstance(raw_reference, Mapping):
                raise StageError("Each reference manifest entry must be an object")
            try:
                metadata = ReferenceMetadata.from_dict(raw_reference["metadata"])
            except (KeyError, ValueError) as exc:
                raise StageError(str(exc)) from exc
            if metadata.slot_id not in slot_ids or metadata.slot_id in entries:
                raise StageError(f"Invalid or duplicate reference slot: {metadata.slot_id}")
            if metadata.asset_path is None or metadata.asset_sha256 is None:
                raise StageError(f"Reference asset metadata is incomplete: {metadata.slot_id}")
            asset_path = (root / metadata.asset_path).resolve()
            if not asset_path.is_relative_to(root):
                raise StageError("Reference asset path escapes the library directory")
            try:
                asset_bytes = asset_path.read_bytes()
            except OSError as exc:
                raise StageError(
                    f"Could not read reference asset {metadata.slot_id}: {exc}"
                ) from exc
            if hashlib.sha256(asset_bytes).hexdigest() != metadata.asset_sha256:
                raise StageError(f"Reference asset checksum failed: {metadata.slot_id}")
            bgr = cv2.imdecode(np.frombuffer(asset_bytes, np.uint8), cv2.IMREAD_COLOR)
            if bgr is None or bgr.shape[:2] != (NORMALIZED_FACE_SIZE, NORMALIZED_FACE_SIZE):
                raise StageError(f"Reference asset has invalid dimensions: {metadata.slot_id}")
            rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            rgb.setflags(write=False)
            raw_landmarks = raw_reference.get("landmarks")
            if not isinstance(raw_landmarks, list) or len(raw_landmarks) != 478:
                raise StageError(f"Reference landmarks are incomplete: {metadata.slot_id}")
            try:
                landmarks = tuple(
                    Point2D(float(point[0]), float(point[1])) for point in raw_landmarks
                )
            except (IndexError, TypeError, ValueError) as exc:
                raise StageError(f"Reference landmarks are invalid: {metadata.slot_id}") from exc
            if not all(isfinite(value) for point in landmarks for value in (point.x, point.y)):
                raise StageError(f"Reference landmarks are non-finite: {metadata.slot_id}")
            entries[metadata.slot_id] = ProcessedReference(metadata, rgb, landmarks)
        return slots, entries

    def close(self) -> None:
        self._preprocessor.close()
