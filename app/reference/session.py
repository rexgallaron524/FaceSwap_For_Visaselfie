"""Persistent reference-library session with automatic save and startup restore."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from app.config import default_app_data_directory
from app.pipeline.types import StageError
from app.reference.library import ReferenceLibraryStore, serialize_manifest
from app.reference.model import ReferenceMetadata

STATE_SCHEMA = "facelive-reference-session"
STATE_VERSION = 1


def default_reference_directory() -> Path:
    return default_app_data_directory() / "references"


class ReferencePersistenceError(StageError):
    """A library edit succeeded in memory but could not be persisted."""


class ReferenceLibrarySession:
    """Own the active library path and persist every successful enrollment edit."""

    def __init__(
        self,
        library: ReferenceLibraryStore | None = None,
        *,
        directory: Path | None = None,
    ) -> None:
        self.library = library or ReferenceLibraryStore()
        self._directory = (directory or default_reference_directory()).resolve()
        self._default_path = self._directory / "default.json"
        self._state_path = self._directory / "active-library.json"
        self._active_path = self._default_path
        self._dirty = False
        self._last_error: str | None = None

    @property
    def active_path(self) -> Path:
        return self._active_path

    @property
    def dirty(self) -> bool:
        return self._dirty

    @property
    def last_error(self) -> str | None:
        return self._last_error

    @property
    def is_saved(self) -> bool:
        return not self._dirty and self._active_path.is_file()

    def restore(self) -> bool:
        """Restore the last active library; return whether a saved library was loaded."""
        self._dirty = False
        self._last_error = None
        had_state = self._state_path.is_file()
        try:
            self._active_path = self._read_active_path()
        except StageError as exc:
            self._active_path = self._default_path
            self._last_error = str(exc)
        if not self._active_path.is_file():
            if had_state:
                self._last_error = (
                    f"Last active reference library was not found: {self._active_path}"
                )
            return False
        try:
            self.library.load(self._active_path)
        except StageError as exc:
            self._last_error = f"Could not restore reference library: {exc}"
            return False
        return True

    def add_reference(self, slot_id: str, image_path: Path) -> ReferenceMetadata:
        metadata = self.library.add_reference(slot_id, image_path)
        self._save_edit("Reference was added")
        return metadata

    def remove_reference(self, slot_id: str) -> None:
        self.library.remove_reference(slot_id)
        self._save_edit("Reference was removed")

    def load(self, path: Path) -> None:
        path = self._json_path(path)
        try:
            self.library.load(path)
        except StageError as exc:
            self._last_error = str(exc)
            raise
        self._activate(path)

    def save_as(self, path: Path) -> Path:
        path = self._json_path(path)
        self.library.save(path)
        self._active_path = path
        self._dirty = False
        try:
            self._write_state(path)
        except StageError as exc:
            self._last_error = str(exc)
            raise ReferencePersistenceError(
                f"The library was saved to {path}, but FaceLive could not remember it: {exc}"
            ) from exc
        self._last_error = None
        return path

    def flush(self) -> None:
        if self._dirty:
            self._save_edit("Reference changes remain in memory")

    def close(self) -> None:
        self.library.close()

    def _save_edit(self, completed_action: str) -> None:
        self._dirty = True
        try:
            self.library.save(self._active_path)
            self._write_state(self._active_path)
        except StageError as exc:
            self._last_error = str(exc)
            raise ReferencePersistenceError(
                f"{completed_action}, but automatic saving failed: {exc}"
            ) from exc
        self._dirty = False
        self._last_error = None

    def _activate(self, path: Path) -> None:
        self._active_path = path
        self._dirty = False
        try:
            self._write_state(path)
        except StageError as exc:
            self._last_error = str(exc)
            raise ReferencePersistenceError(
                f"The library was loaded, but FaceLive could not remember it: {exc}"
            ) from exc
        self._last_error = None

    def _read_active_path(self) -> Path:
        if not self._state_path.is_file():
            return self._default_path
        try:
            if self._state_path.stat().st_size > 64 * 1024:
                raise StageError("Reference session file is unexpectedly large")
            value = json.loads(self._state_path.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                raise ValueError("root must be an object")
            if value.get("schema") != STATE_SCHEMA or value.get("version") != STATE_VERSION:
                raise ValueError("unsupported schema or version")
            raw_path = value.get("active_library")
            if not isinstance(raw_path, str) or not raw_path.strip():
                raise ValueError("active library path is missing")
            path = Path(raw_path).expanduser()
            if not path.is_absolute():
                raise ValueError("active library path must be absolute")
            return path.resolve()
        except StageError:
            raise
        except (OSError, ValueError, TypeError) as exc:
            raise StageError(f"Could not read reference session: {exc}") from exc

    def _write_state(self, active_path: Path) -> None:
        value: dict[str, Any] = {
            "schema": STATE_SCHEMA,
            "version": STATE_VERSION,
            "active_library": str(active_path.resolve()),
        }
        temporary = self._state_path.with_name(f".{self._state_path.name}.{os.getpid()}.tmp")
        try:
            self._directory.mkdir(parents=True, exist_ok=True)
            temporary.write_text(serialize_manifest(value), encoding="utf-8", newline="\n")
            temporary.replace(self._state_path)
        except OSError as exc:
            temporary.unlink(missing_ok=True)
            raise StageError(f"Could not save reference session: {exc}") from exc

    @staticmethod
    def _json_path(path: Path) -> Path:
        path = path.expanduser().resolve()
        return path if path.suffix.casefold() == ".json" else path.with_suffix(".json")
