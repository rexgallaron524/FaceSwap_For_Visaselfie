from typing import Protocol

from app.pipeline.types import RenderedFace, VideoFrame


class Compositor(Protocol):
    def composite(self, original: VideoFrame, face: RenderedFace) -> VideoFrame:
        """Return a new RGB frame with the original ID/time and unchanged pixels

        wherever alpha is zero. Mismatched ID/time/size raises StageError.
        Inputs must never be mutated.
        """
        ...
