from __future__ import annotations

import multiprocessing
import os
import time
from queue import Empty
from uuid import uuid4

import numpy as np
import pytest

from app.pipeline.types import FrameFormat, StageError, VideoFrame
from app.transport import SharedMemoryFrameConsumer, SharedMemoryFrameSink, TransportState
from app.transport.shared_memory import (
    HEADER_SIZE,
    HEADER_STRUCT,
    SLOT_HEADER_SIZE,
    SLOT_STRUCT,
    slot_stride,
    transport_size,
)

pytestmark = pytest.mark.skipif(os.name != "nt", reason="Windows named-mutex transport")


def _name() -> str:
    return f"facelive_test_{uuid4().hex}"


def _frame(frame_id: int, width: int = 32, height: int = 18) -> VideoFrame:
    pixels = np.full((height, width, 3), frame_id % 251, dtype=np.uint8)
    pixels.setflags(write=False)
    return VideoFrame(frame_id, 1_000_000_000 + frame_id, pixels)


def _stress_consumer(name, ready, finished, results) -> None:
    consumer = SharedMemoryFrameConsumer(name)
    try:
        deadline = time.monotonic() + 5
        while True:
            try:
                consumer.open()
                break
            except StageError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.005)
        consumer.read_latest()  # Register the heartbeat before the producer starts.
        ready.set()
        received = 0
        last_sequence = 0
        errors: list[str] = []
        drain_deadline = None
        while True:
            frame = consumer.read_latest()
            if frame is not None:
                received += 1
                if frame.sequence <= last_sequence:
                    errors.append("sequence did not increase")
                if not np.all(frame.rgb == frame.frame_id % 251):
                    errors.append("partially written frame became visible")
                last_sequence = frame.sequence
            if finished.is_set():
                if drain_deadline is None:
                    drain_deadline = time.monotonic() + 0.25
                elif time.monotonic() >= drain_deadline:
                    break
            time.sleep(0.0002)
        results.put((received, last_sequence, consumer.skipped_frames, errors))
    except Exception as exc:
        results.put((0, 0, 0, [repr(exc)]))
    finally:
        consumer.close()


def _crash_producer(name, ready) -> None:
    sink = SharedMemoryFrameSink(
        name,
        capacity_width=32,
        capacity_height=18,
        require_consumer=False,
    )
    sink.open(FrameFormat(32, 18, 30))
    sink.publish(_frame(1))
    ready.set()
    time.sleep(30)


def test_binary_layout_sizes_and_capacity_are_stable():
    assert HEADER_STRUCT.size == HEADER_SIZE == 256
    assert SLOT_STRUCT.size == SLOT_HEADER_SIZE == 64
    assert slot_stride(32, 18) % 64 == 0
    assert transport_size(32, 18, 3) == HEADER_SIZE + slot_stride(32, 18) * 3


def test_sink_requires_a_consumer_and_consumer_reads_only_the_newest_frame():
    name = _name()
    sink = SharedMemoryFrameSink(name, capacity_width=32, capacity_height=18)
    consumer = SharedMemoryFrameConsumer(name)
    try:
        sink.open(FrameFormat(32, 18, 30))
        assert sink.publish(_frame(0)) is False
        consumer.open()
        assert consumer.status().producer_alive is True
        assert consumer.read_latest() is None

        assert all(sink.publish(_frame(frame_id)) for frame_id in range(1, 7))
        newest = consumer.read_latest()

        assert newest is not None
        assert (newest.sequence, newest.frame_id) == (6, 6)
        assert np.all(newest.rgb == 6)
        assert consumer.skipped_frames == 0  # The first observed frame establishes the baseline.
        assert sink.status().published_count == 6
        assert sink.status().dropped_count == 1
        assert consumer.read_latest() is None
    finally:
        sink.close()
        consumer.close()


def test_clean_restart_changes_session_and_resets_sequence_without_remapping_consumer():
    name = _name()
    first = SharedMemoryFrameSink(name, capacity_width=32, capacity_height=18)
    consumer = SharedMemoryFrameConsumer(name)
    second = SharedMemoryFrameSink(name, capacity_width=32, capacity_height=18)
    try:
        first.open(FrameFormat(32, 18, 30))
        consumer.open()
        consumer.read_latest()
        assert first.publish(_frame(1))
        before = consumer.read_latest()
        assert before is not None
        first.close()
        assert consumer.status().state is TransportState.STOPPED
        assert consumer.status().producer_alive is False

        second.open(FrameFormat(32, 18, 30))
        consumer.read_latest()  # Register with the new session.
        assert second.publish(_frame(2))
        after = consumer.read_latest()

        assert after is not None
        assert after.session_id != before.session_id
        assert (after.sequence, after.frame_id) == (1, 2)
        assert second.status().restart_count == 1
    finally:
        first.close()
        second.close()
        consumer.close()


def test_second_live_producer_is_rejected():
    name = _name()
    first = SharedMemoryFrameSink(name, capacity_width=32, capacity_height=18)
    second = SharedMemoryFrameSink(name, capacity_width=32, capacity_height=18)
    try:
        first.open(FrameFormat(32, 18, 30))
        with pytest.raises(StageError, match="live producer"):
            second.open(FrameFormat(32, 18, 30))
    finally:
        first.close()
        second.close()


def test_second_live_consumer_is_rejected():
    name = _name()
    sink = SharedMemoryFrameSink(name, capacity_width=32, capacity_height=18)
    first = SharedMemoryFrameConsumer(name)
    second = SharedMemoryFrameConsumer(name)
    try:
        sink.open(FrameFormat(32, 18, 30))
        first.open()
        with pytest.raises(StageError, match="live consumer"):
            second.open()
    finally:
        first.close()
        second.close()
        sink.close()


def test_cross_process_stress_never_exposes_partial_frames():
    name = _name()
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    finished = context.Event()
    results = context.Queue()
    sink = SharedMemoryFrameSink(name, capacity_width=32, capacity_height=18)
    process = context.Process(target=_stress_consumer, args=(name, ready, finished, results))
    try:
        sink.open(FrameFormat(32, 18, 30))
        process.start()
        assert ready.wait(5)
        accepted = 0
        for frame_id in range(1, 1_001):
            accepted += sink.publish(_frame(frame_id))
        finished.set()
        process.join(10)
        assert process.exitcode == 0
        received, last_sequence, skipped, errors = results.get(timeout=2)
        assert errors == []
        assert accepted > 0
        assert received > 0
        assert 0 < last_sequence <= accepted
        assert skipped >= 0
        status = sink.status()
        assert status.published_count == accepted
        assert status.published_count + status.dropped_count >= 1_000
    finally:
        finished.set()
        if process.is_alive():
            process.terminate()
        process.join(2)
        sink.close()
        try:
            results.get_nowait()
        except Empty:
            pass


def test_dead_producer_can_be_replaced_while_consumer_keeps_mapping_open():
    name = _name()
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    process = context.Process(target=_crash_producer, args=(name, ready))
    consumer = SharedMemoryFrameConsumer(name)
    replacement = SharedMemoryFrameSink(
        name,
        capacity_width=32,
        capacity_height=18,
        require_consumer=False,
    )
    try:
        process.start()
        assert ready.wait(5)
        consumer.open()
        original = consumer.read_latest()
        assert original is not None
        process.terminate()
        process.join(5)
        assert process.exitcode is not None

        replacement.open(FrameFormat(32, 18, 30))
        assert replacement.publish(_frame(2))
        current = consumer.read_latest()

        assert current is not None
        assert current.session_id != original.session_id
        assert (current.sequence, current.frame_id) == (1, 2)
    finally:
        if process.is_alive():
            process.terminate()
        process.join(2)
        replacement.close()
        consumer.close()
