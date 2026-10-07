from typing import Protocol

from app.pipeline.types import FrameFormat, VideoFrame


class FrameSink(Protocol):
    def open(self, frame_format: FrameFormat) -> None:
        """Prepare output resources, or raise StageError."""
        ...

    def publish(self, frame: VideoFrame) -> bool:
        """Nonblocking: True when accepted, False when dropped/no consumer.

        Keep only bounded complete frames; never expose partially written data.
        Sink failure raises StageError. The caller owns output/fallback decisions.
        """
        ...

    def close(self) -> None:
        """Idempotently stop output and release resources."""
        ...
