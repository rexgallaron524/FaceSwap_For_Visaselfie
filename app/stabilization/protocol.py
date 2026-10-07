from typing import Protocol

from app.pipeline.types import FaceState


class Stabilizer(Protocol):
    def update(self, face: FaceState) -> FaceState:
        """Smooth pose/geometry/expression preserving frame ID/time and units."""
        ...

    def reset(self) -> None:
        """Clear history after tracking loss, source/library change, or session restart."""
        ...
