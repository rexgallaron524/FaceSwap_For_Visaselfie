"""Small backend-independent capture metrics."""

from collections import deque

from app.pipeline.types import VideoFrame


class CaptureMetrics:
    def __init__(self, window_seconds: float = 2.0) -> None:
        if window_seconds <= 0:
            raise ValueError("window_seconds must be positive")
        self._window_ns = int(window_seconds * 1_000_000_000)
        self._samples: deque[tuple[int, int]] = deque()
        self.preview_drops = 0

    def reset(self) -> None:
        self._samples.clear()
        self.preview_drops = 0

    def observe(self, frame: VideoFrame) -> None:
        if self._samples:
            previous_id = self._samples[-1][0]
            if frame.frame_id > previous_id:
                self.preview_drops += max(0, frame.frame_id - previous_id - 1)
            elif frame.frame_id <= previous_id:
                self.reset()
        self._samples.append((frame.frame_id, frame.timestamp_ns))
        cutoff = frame.timestamp_ns - self._window_ns
        while len(self._samples) > 2 and self._samples[0][1] < cutoff:
            self._samples.popleft()

    @property
    def fps(self) -> float | None:
        if len(self._samples) < 2:
            return None
        first_id, first_time = self._samples[0]
        last_id, last_time = self._samples[-1]
        elapsed = (last_time - first_time) / 1_000_000_000
        if elapsed <= 0 or last_id <= first_id:
            return None
        return (last_id - first_id) / elapsed
