"""Stress the shared-memory frame ring with a separate consumer process."""

from __future__ import annotations

import argparse
import multiprocessing
import statistics
import time
from queue import Empty
from uuid import uuid4

import numpy as np

from app.pipeline.types import FrameFormat, StageError, VideoFrame
from app.transport import SharedMemoryFrameConsumer, SharedMemoryFrameSink
from app.transport.shared_memory import transport_size


def consume(name, ready, finished, results, delay_ms: float) -> None:
    consumer = SharedMemoryFrameConsumer(name)
    received = 0
    last_sequence = 0
    corrupt = 0
    ages: list[float] = []
    try:
        deadline = time.monotonic() + 10
        while True:
            try:
                consumer.open()
                break
            except StageError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.01)
        consumer.read_latest()
        ready.set()
        drain_until = None
        while True:
            frame = consumer.read_latest()
            if frame is not None:
                received += 1
                last_sequence = frame.sequence
                expected = frame.frame_id % 251
                if not np.all(frame.rgb == expected):
                    corrupt += 1
                ages.append((time.monotonic_ns() - frame.timestamp_ns) / 1_000_000)
            if finished.is_set():
                if drain_until is None:
                    drain_until = time.monotonic() + 0.25
                elif time.monotonic() >= drain_until:
                    break
            if delay_ms:
                time.sleep(delay_ms / 1000)
        results.put(
            {
                "received": received,
                "last_sequence": last_sequence,
                "skipped": consumer.skipped_frames,
                "corrupt": corrupt,
                "age_mean": statistics.fmean(ages) if ages else 0.0,
                "age_p95": float(np.percentile(ages, 95)) if ages else 0.0,
            }
        )
    except Exception as exc:
        results.put({"error": repr(exc)})
    finally:
        consumer.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--frames", type=int, default=600)
    parser.add_argument("--consumer-delay-ms", type=float, default=1.0)
    args = parser.parse_args()
    if args.width <= 0 or args.height <= 0 or args.frames <= 0 or args.consumer_delay_ms < 0:
        parser.error("dimensions/frames must be positive and consumer delay nonnegative")

    name = f"facelive_benchmark_{uuid4().hex}"
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    finished = context.Event()
    results = context.Queue()
    process = context.Process(
        target=consume,
        args=(name, ready, finished, results, args.consumer_delay_ms),
    )
    sink = SharedMemoryFrameSink(
        name,
        capacity_width=args.width,
        capacity_height=args.height,
    )
    pixels = np.empty((args.height, args.width, 3), dtype=np.uint8)
    pixels.setflags(write=False)
    accepted = 0
    publish_times: list[float] = []
    started = time.perf_counter()
    try:
        sink.open(FrameFormat(args.width, args.height, 30))
        process.start()
        if not ready.wait(10):
            raise RuntimeError("Consumer did not attach")
        started = time.perf_counter()
        for frame_id in range(args.frames):
            pixels.setflags(write=True)
            pixels.fill(frame_id % 251)
            pixels.setflags(write=False)
            frame = VideoFrame(frame_id, time.monotonic_ns(), pixels)
            before = time.perf_counter_ns()
            accepted += sink.publish(frame)
            publish_times.append((time.perf_counter_ns() - before) / 1_000_000)
        elapsed = time.perf_counter() - started
        finished.set()
        process.join(15)
        if process.exitcode != 0:
            raise RuntimeError(f"Consumer exited with {process.exitcode}")
        try:
            result = results.get(timeout=2)
        except Empty as exc:
            raise RuntimeError("Consumer did not return results") from exc
        if "error" in result:
            raise RuntimeError(result["error"])
        status = sink.status()
        print(f"Format: {args.width}x{args.height} RGB24; slots: {sink.slot_count}")
        print(
            f"Producer attempts: {args.frames}; accepted: {accepted}; "
            f"dropped: {status.dropped_count}"
        )
        print(
            f"Producer rate: {args.frames / elapsed:.2f} attempts/s; "
            f"accepted rate: {accepted / elapsed:.2f} frames/s"
        )
        print(
            f"Publish copy: mean {statistics.fmean(publish_times):.2f} ms; "
            f"p95 {float(np.percentile(publish_times, 95)):.2f} ms"
        )
        print(
            f"Consumer received: {result['received']}; newest sequence: "
            f"{result['last_sequence']}; skipped: {result['skipped']}"
        )
        print(
            f"Frame age: mean {result['age_mean']:.2f} ms; "
            f"p95 {result['age_p95']:.2f} ms; corrupt: {result['corrupt']}"
        )
        print(
            f"Shared-memory size: {transport_size(args.width, args.height, sink.slot_count)} bytes"
        )
        return 0 if result["corrupt"] == 0 else 1
    finally:
        finished.set()
        if process.is_alive():
            process.terminate()
        process.join(2)
        sink.close()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
