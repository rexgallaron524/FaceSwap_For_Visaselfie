from pathlib import Path
from typing import Protocol

from app.pipeline.types import FaceState, PreparedReference, ReferenceWeight


class ReferenceLibrary(Protocol):
    def load(self, path: Path) -> None:
        """Validate/align/cache a library once, or raise StageError.

        Failed loads leave the library empty; never silently retain an old subject.
        """
        ...

    def references(self) -> tuple[PreparedReference, ...]:
        """Return cached references; empty means no library loaded."""
        ...

    def clear(self) -> None:
        """Idempotently release cached references."""
        ...


class ReferenceSelector(Protocol):
    def select(
        self, face: FaceState, references: tuple[PreparedReference, ...]
    ) -> tuple[ReferenceWeight, ...]:
        """Return continuous normalized weights by yaw/pitch/expression.

        Empty means unsupported pose/no compatible references. Roll and scale are geometric.
        """
        ...
