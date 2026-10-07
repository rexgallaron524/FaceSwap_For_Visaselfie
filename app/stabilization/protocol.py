from typing import Protocol

from app.pipeline.types import FaceState, ReferenceWeight


class Stabilizer(Protocol):
    def update(self, face: FaceState) -> FaceState:
        """Smooth pose/geometry/expression preserving frame ID/time and units."""
        ...

    def smooth_weights(
        self, weights: tuple[ReferenceWeight, ...], timestamp_ns: int
    ) -> tuple[ReferenceWeight, ...]:
        """Smooth a normalized selection while preserving normalization."""
        ...

    def coast(self, frame_id: int, timestamp_ns: int) -> FaceState | None:
        """Return a bounded held state for a brief tracking gap, or None."""
        ...

    def reset(self) -> None:
        """Clear history after tracking loss, source/library change, or session restart."""
        ...
