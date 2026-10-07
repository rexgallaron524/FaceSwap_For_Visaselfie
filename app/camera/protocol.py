from typing import Protocol

from app.pipeline.types import CameraDevice, FrameFormat, VideoFrame


class CameraSource(Protocol):
    def enumerate_devices(self) -> tuple[CameraDevice, ...]:
        """List devices without opening them; identify virtual inputs explicitly."""
        ...

    def open(self, device_id: str, requested: FrameFormat) -> FrameFormat:
        """Open a selected physical device and return negotiated format; or StageError."""
        ...

    def read_latest(self) -> VideoFrame | None:
        """Nonblocking: consume newest complete frame, or None if none is new.

        Disconnect/failure raises StageError. Old pending frames are dropped.
        """
        ...

    def close(self) -> None:
        """Idempotently release device and worker resources."""
        ...
