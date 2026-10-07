from typing import Protocol

from app.pipeline.types import (
    FaceState,
    FrameFormat,
    PreparedReference,
    ReferenceWeight,
    RenderedFace,
)


class FaceRenderer(Protocol):
    def open(self, output_format: FrameFormat) -> None:
        """Allocate renderer resources, or raise StageError."""
        ...

    def render(
        self,
        face: FaceState,
        references: tuple[PreparedReference, ...],
        weights: tuple[ReferenceWeight, ...],
    ) -> RenderedFace:
        """Render the face/mask matching face frame ID/time; failures raise StageError."""
        ...

    def close(self) -> None:
        """Idempotently release renderer resources."""
        ...
