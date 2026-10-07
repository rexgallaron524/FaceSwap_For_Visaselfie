from pathlib import Path

import numpy as np
import pytest

from app.pipeline.types import Point2D, Rect
from app.reference.library import ReferenceLibraryStore, parse_manifest
from app.reference.model import ProcessedReference, ReferenceMetadata
from app.reference.session import ReferenceLibrarySession, ReferencePersistenceError


class FakePreprocessor:
    def process(self, slot, path):
        rgb = np.full((512, 512, 3), 80, dtype=np.uint8)
        rgb.setflags(write=False)
        return ProcessedReference(
            ReferenceMetadata(
                slot_id=slot.slot_id,
                source_name=path.name,
                source_sha256="a" * 64,
                original_width=800,
                original_height=800,
                face_bounds=Rect(0.2, 0.2, 0.5, 0.5),
                brightness=100.0,
                sharpness=50.0,
                face_coverage=0.25,
                alignment_rotation_degrees=0.0,
                created_at="2026-10-07T00:00:00+00:00",
            ),
            rgb,
            tuple(Point2D(float(index), float(index)) for index in range(478)),
        )

    def close(self):
        pass


def make_session(directory: Path) -> ReferenceLibrarySession:
    library = ReferenceLibraryStore(FakePreprocessor(), clock=lambda: "2026-10-07T01:00:00+00:00")
    return ReferenceLibrarySession(library, directory=directory)


def test_enrollment_auto_saves_and_restores_after_restart(tmp_path):
    directory = tmp_path / "references"
    session = make_session(directory)
    assert not session.restore()

    session.add_reference("front-neutral", tmp_path / "front.jpg")

    assert session.is_saved
    assert not session.dirty
    assert session.active_path == directory / "default.json"
    assert session.active_path.is_file()
    assert len(tuple((directory / "default_assets").glob("front-neutral-*.png"))) == 1

    restarted = make_session(directory)
    assert restarted.restore()
    assert restarted.library.reference("front-neutral") is not None
    assert restarted.active_path == session.active_path


def test_save_as_becomes_active_and_future_edits_use_that_library(tmp_path):
    directory = tmp_path / "references"
    export = tmp_path / "named-library.json"
    session = make_session(directory)
    session.add_reference("front-neutral", tmp_path / "front.jpg")

    assert session.save_as(export) == export
    session.add_reference("front-smile", tmp_path / "smile.jpg")

    manifest = parse_manifest(export.read_text(encoding="utf-8"))
    saved_ids = [item["metadata"]["slot_id"] for item in manifest["references"]]
    assert saved_ids == ["front-neutral", "front-smile"]
    restarted = make_session(directory)
    assert restarted.restore()
    assert restarted.active_path == export
    assert restarted.library.valid_count == 2


def test_removing_last_reference_auto_saves_empty_library(tmp_path):
    directory = tmp_path / "references"
    session = make_session(directory)
    session.add_reference("front-neutral", tmp_path / "front.jpg")

    session.remove_reference("front-neutral")

    assert session.is_saved
    restarted = make_session(directory)
    assert restarted.restore()
    assert restarted.library.references() == ()
    assert not tuple((directory / "default_assets").glob("front-neutral-*.png"))


def test_auto_save_failure_keeps_in_memory_edit_marked_dirty(tmp_path):
    blocked_directory = tmp_path / "not-a-directory"
    blocked_directory.write_text("blocked", encoding="utf-8")
    session = make_session(blocked_directory)

    with pytest.raises(ReferencePersistenceError, match="automatic saving failed"):
        session.add_reference("front-neutral", tmp_path / "front.jpg")

    assert session.library.reference("front-neutral") is not None
    assert session.dirty
    assert session.last_error is not None
