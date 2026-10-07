"""Standalone newest-frame consumer for the FaceLive shared-memory transport."""

from __future__ import annotations

import argparse
import statistics
import time

import numpy as np

from app.pipeline.types import StageError
from app.transport import SharedMemoryFrameConsumer, TransportState


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="facelive_frames_v1")
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--poll-ms", type=float, default=1.0)
    parser.add_argument("--attach-timeout", type=float, default=10.0)
    args = parser.parse_args()
    if args.seconds <= 0 or args.poll_ms < 0 or args.attach_timeout <= 0:
        parser.error("seconds/attach-timeout must be positive and poll-ms nonnegative")

    consumer: SharedMemoryFrameConsumer | None = None
    attach_deadline = time.monotonic() + args.attach_timeout
    end_at: float | None = None
    frames = 0
    sessions = set()
    latencies: list[float] = []
    first_sequence = 0
    last_sequence = 0
    reconnects = 0
    skipped = 0
    read_errors: list[str] = []
    try:
        while end_at is None or time.monotonic() < end_at:
            if consumer is None:
                candidate = SharedMemoryFrameConsumer(args.name)
                try:
                    candidate.open()
                except StageError:
                    candidate.close()
                    if time.monotonic() >= attach_deadline:
                        print(f"Could not attach to transport '{args.name}'")
                        return 2
                    time.sleep(0.05)
                    continue
                consumer = candidate
                reconnects += 1
                if end_at is None:
                    end_at = time.monotonic() + args.seconds
            try:
                frame = consumer.read_latest()
                status = consumer.status()
            except StageError as exc:
                skipped += consumer.skipped_frames
                read_errors.append(str(exc))
                consumer.close()
                consumer = None
                time.sleep(0.02)
                continue
            if frame is not None:
                frames += 1
                sessions.add(frame.session_id)
                first_sequence = first_sequence or frame.sequence
                last_sequence = frame.sequence
                age_ms = (time.monotonic_ns() - frame.timestamp_ns) / 1_000_000
                if np.isfinite(age_ms) and age_ms >= 0:
                    latencies.append(age_ms)
            if status.state in (TransportState.STOPPED, TransportState.ERROR):
                skipped += consumer.skipped_frames
                consumer.close()
                consumer = None
            if args.poll_ms:
                time.sleep(args.poll_ms / 1000)
    except KeyboardInterrupt:
        pass
    finally:
        if consumer is not None:
            skipped += consumer.skipped_frames
            consumer.close()

    print(f"Transport: {args.name}")
    print(f"Frames received: {frames}; sessions: {len(sessions)}; reconnects: {reconnects}")
    print(f"Sequence range: {first_sequence}..{last_sequence}; stale frames skipped: {skipped}")
    if latencies:
        print(
            f"Frame age: mean {statistics.fmean(latencies):.2f} ms; "
            f"p95 {float(np.percentile(latencies, 95)):.2f} ms"
        )
    if read_errors:
        print(f"Read errors: {len(read_errors)}; last: {read_errors[-1]}")
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
