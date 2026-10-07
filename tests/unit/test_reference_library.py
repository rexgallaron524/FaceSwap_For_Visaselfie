from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from app.pipeline.types import Point2D, Rect, StageError
from app.reference.library import ReferenceLibraryStore, parse_manifest, serialize_manifest
from app.reference.model import (
    INITIAL_REFERENCE_SLOTS,
    ProcessedReference,
    ReferenceMetadata,
    ReferenceSlot,
)


class FakePreprocessor:
    def __init__(self):
        self.closed = False

    def process(self, slot, path):
        rgb = np.zeros((512, 512, 3), dtype=np.uint8)
        rgb[:, :, 0] = 31
        rgb.setflags(write=False)
        metadata = ReferenceMetadata(
            slot_id=slot.slot_id,
            source_name=path.name,
            source_sha256="a" * 64,
            original_width=1000,
            original_height=800,
            face_bounds=Rect(0.2, 0.1, 0.5, 0.7),
            brightness=120.0,
            sharpness=90.0,
            face_coverage=0.35,
            alignment_rotation_degrees=1.5,
            created_at="2026-10-07T00:00:00+00:00",
            blendshapes={"mouthSmileLeft": 0.4},
        )
        landmarks = tuple(Point2D(float(index), float(index + 1)) for index in range(478))
        return ProcessedReference(metadata, rgb, landmarks)

    def close(self):
        self.closed = True


def test_metadata_and_manifest_serialization_are_deterministic():
    processor = FakePreprocessor()
    store = ReferenceLibraryStore(processor, clock=lambda: "2026-10-07T01:00:00+00:00")
    metadata = store.add_reference("front-neutral", Path("subject.png"))

    round_trip = ReferenceMetadata.from_dict(metadata.to_dict())
    assert round_trip == metadata
    payload = {"z": 1, "metadata": metadata.to_dict(), "a": "é"}
    assert serialize_manifest(payload) == serialize_manifest(payload)
    assert list(parse_manifest(serialize_manifest(payload))) == ["a", "metadata", "z"]


def test_library_saves_loads_cached_assets_and_looks_up_by_slot(tmp_path):
    store = ReferenceLibraryStore(FakePreprocessor(), clock=lambda: "2026-10-07T01:00:00+00:00")
    store.add_reference("right-40", tmp_path / "right.jpg")
    store.add_reference("front-neutral", tmp_path / "front.jpg")
    path = tmp_path / "library.json"

    store.save(path)
    loaded = ReferenceLibraryStore(FakePreprocessor())
    loaded.load(path)

    assert loaded.valid_count == 2
    assert loaded.required_count == 8
    assert not loaded.is_complete
    assert loaded.reference("front-neutral").reference_id == "front-neutral"
    assert loaded.reference("left-20") is None
    assert [reference.reference_id for reference in loaded.references()] == [
        "front-neutral",
        "right-40",
    ]
    assert not loaded.reference("front-neutral").rgb.flags.writeable
    assert len(loaded.reference("right-40").landmarks) == 478
    first_text = path.read_text(encoding="utf-8")
    store.save(path)
    assert path.read_text(encoding="utf-8") == first_text


def test_failed_load_clears_existing_subject_and_detects_corruption(tmp_path):
    store = ReferenceLibraryStore(FakePreprocessor())
    store.add_reference("front-neutral", tmp_path / "front.jpg")
    path = tmp_path / "library.json"
    store.save(path)
    manifest = parse_manifest(path.read_text(encoding="utf-8"))
    asset = tmp_path / manifest["references"][0]["metadata"]["asset"]
    asset.write_bytes(asset.read_bytes() + b"corrupt")

    with pytest.raises(StageError, match="checksum failed"):
        store.load(path)
    assert store.references() == ()
    assert store.manifest_path is None


def test_saving_after_removal_deletes_the_cached_asset(tmp_path):
    store = ReferenceLibraryStore(FakePreprocessor())
    store.add_reference("front-neutral", tmp_path / "front.jpg")
    path = tmp_path / "library.json"
    store.save(path)
    manifest = parse_manifest(path.read_text(encoding="utf-8"))
    asset = tmp_path / manifest["references"][0]["metadata"]["asset"]
    assert asset.is_file()

    store.add_reference("front-smile", tmp_path / "smile.jpg")
    store.remove_reference("front-neutral")
    store.save(path)

    assert not asset.exists()
    loaded = ReferenceLibraryStore(FakePreprocessor())
    loaded.load(path)
    assert loaded.reference("front-neutral") is None
    assert loaded.reference("front-smile") is not None


def test_slot_model_supports_additional_pose_variants():
    store = ReferenceLibraryStore(FakePreprocessor())
    custom = ReferenceSlot(
        "left-smile-20",
        "Left smile 20°",
        20.0,
        0.0,
        "smile",
        "Turn left and smile.",
        required=False,
    )
    store.add_slot(custom)
    assert store.slot("left-smile-20") == custom
    assert store.required_count == len(INITIAL_REFERENCE_SLOTS)


def test_unknown_lookup_and_duplicate_slot_are_rejected():
    store = ReferenceLibraryStore(FakePreprocessor())
    with pytest.raises(StageError, match="Unknown reference slot"):
        store.add_reference("missing", Path("missing.png"))
    with pytest.raises(ValueError, match="already exists"):
        store.add_slot(replace(INITIAL_REFERENCE_SLOTS[0]))
