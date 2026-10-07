from typing import Protocol

from app.pipeline.types import TrackingResult, VideoFrame


class FaceTracker(Protocol):
    def open(self) -> None:
        """Load backend resources, or raise StageError."""
        ...

    def submit(self, frame: VideoFrame) -> bool:
        """Nonblocking: True if accepted, False if busy and dropped.

        Accepted frames retain their IDs and timestamps through async processing.
        Inputs must have strictly increasing IDs/timestamps within a session.
        """
        ...

    def poll_latest(self) -> TrackingResult | None:
        """Consume newest completed result; None means pending, distinct from NO_FACE."""
        ...

    def close(self) -> None:
        """Idempotently stop callbacks and release model/worker resources."""
        ...
