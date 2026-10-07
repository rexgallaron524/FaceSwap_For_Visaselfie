"""Versioned Windows shared-memory ring for complete RGB video frames.

The byte layout is a native ABI. Keep it synchronized with
docs/frame_transport_protocol.md and the future C++ declarations.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import struct
import threading
import time
import zlib
from dataclasses import dataclass
from enum import IntEnum
from fractions import Fraction
from multiprocessing import shared_memory
from uuid import UUID, uuid4

import numpy as np

from app.pipeline.types import FrameFormat, StageError, VideoFrame

MAGIC = b"FLVCAM01"
PROTOCOL_VERSION = 1
ENDIAN_MARKER = 0x01020304
HEADER_SIZE = 256
SLOT_HEADER_SIZE = 64
DEFAULT_SLOT_COUNT = 3
MAX_SLOT_COUNT = 8

# Header: 176 bytes of fields followed by 80 reserved zero bytes.
HEADER_STRUCT = struct.Struct("<8sHHIQQQ10IQ4I5Q2I3Q80x")
SLOT_STRUCT = struct.Struct("<QQQQQ6I")

_OFF_PRODUCER_PID = 88
_OFF_CONSUMER_PID = 92
_OFF_STATE = 96
_OFF_ERROR_CODE = 100
_OFF_PRODUCER_HEARTBEAT = 104
_OFF_CONSUMER_HEARTBEAT = 112
_OFF_CONSUMER_SESSION_HIGH = 120
_OFF_CONSUMER_SESSION_LOW = 128
_OFF_PUBLISHED_SEQUENCE = 136
_OFF_PUBLISHED_SLOT = 144
_OFF_PUBLISHED_COUNT = 152
_OFF_DROPPED_COUNT = 160
_OFF_RESTART_COUNT = 168

_WAIT_OBJECT_0 = 0
_WAIT_ABANDONED = 0x80
_WAIT_TIMEOUT = 0x102
_SYNCHRONIZE = 0x00100000


class PixelFormat(IntEnum):
    RGB24 = 1


class TransportState(IntEnum):
    INITIALIZING = 1
    RUNNING = 2
    STOPPED = 3
    ERROR = 4


@dataclass(frozen=True, slots=True)
class TransportFrame:
    session_id: UUID
    sequence: int
    frame_id: int
    timestamp_ns: int
    width: int
    height: int
    stride: int
    pixel_format: PixelFormat
    rgb: np.ndarray


@dataclass(frozen=True, slots=True)
class TransportStatus:
    session_id: UUID
    state: TransportState
    producer_pid: int
    consumer_pid: int
    producer_heartbeat_ns: int
    consumer_heartbeat_ns: int
    published_sequence: int
    published_count: int
    dropped_count: int
    restart_count: int
    producer_alive: bool
    consumer_connected: bool
    error_code: int


@dataclass(frozen=True, slots=True)
class _Header:
    total_size: int
    session_high: int
    session_low: int
    capacity_width: int
    capacity_height: int
    width: int
    height: int
    stride: int
    pixel_format: int
    fps_numerator: int
    fps_denominator: int
    slot_count: int
    slot_header_size: int
    slot_stride: int
    producer_pid: int
    consumer_pid: int
    state: int
    error_code: int
    producer_heartbeat_ns: int
    consumer_heartbeat_ns: int
    consumer_session_high: int
    consumer_session_low: int
    published_sequence: int
    published_slot: int
    published_count: int
    dropped_count: int
    restart_count: int

    @property
    def session_id(self) -> UUID:
        return UUID(int=(self.session_high << 64) | self.session_low)


def _align(value: int, alignment: int = 64) -> int:
    return (value + alignment - 1) // alignment * alignment


def slot_stride(capacity_width: int, capacity_height: int) -> int:
    return _align(SLOT_HEADER_SIZE + capacity_width * capacity_height * 3)


def transport_size(capacity_width: int, capacity_height: int, slot_count: int) -> int:
    return HEADER_SIZE + slot_stride(capacity_width, capacity_height) * slot_count


def _split_uuid(value: UUID) -> tuple[int, int]:
    return value.int >> 64, value.int & ((1 << 64) - 1)


def _read_header(buffer: memoryview) -> _Header:
    values = HEADER_STRUCT.unpack_from(buffer, 0)
    magic, version, header_size, endian = values[:4]
    if magic != MAGIC:
        raise StageError("Frame transport magic is invalid")
    if version != PROTOCOL_VERSION:
        raise StageError(f"Unsupported frame transport version: {version}")
    if header_size != HEADER_SIZE or endian != ENDIAN_MARKER:
        raise StageError("Frame transport header layout is incompatible")
    (
        total_size,
        session_high,
        session_low,
        capacity_width,
        capacity_height,
        width,
        height,
        stride,
        pixel_format,
        fps_numerator,
        fps_denominator,
        slot_count,
        slot_header_size,
        slot_stride_value,
        producer_pid,
        consumer_pid,
        state,
        error_code,
        producer_heartbeat_ns,
        consumer_heartbeat_ns,
        consumer_session_high,
        consumer_session_low,
        published_sequence,
        published_slot,
        _reserved,
        published_count,
        dropped_count,
        restart_count,
    ) = values[4:]
    return _Header(
        total_size,
        session_high,
        session_low,
        capacity_width,
        capacity_height,
        width,
        height,
        stride,
        pixel_format,
        fps_numerator,
        fps_denominator,
        slot_count,
        slot_header_size,
        slot_stride_value,
        producer_pid,
        consumer_pid,
        state,
        error_code,
        producer_heartbeat_ns,
        consumer_heartbeat_ns,
        consumer_session_high,
        consumer_session_low,
        published_sequence,
        published_slot,
        published_count,
        dropped_count,
        restart_count,
    )


def _validate_header(header: _Header, actual_size: int) -> None:
    if header.total_size > actual_size:
        raise StageError("Frame transport is smaller than its declared layout")
    if header.slot_count < 2 or header.slot_count > MAX_SLOT_COUNT:
        raise StageError("Frame transport slot count is invalid")
    if header.slot_header_size != SLOT_HEADER_SIZE:
        raise StageError("Frame transport slot header is incompatible")
    expected_stride = slot_stride(header.capacity_width, header.capacity_height)
    if header.slot_stride != expected_stride:
        raise StageError("Frame transport slot stride is incompatible")
    if (
        transport_size(header.capacity_width, header.capacity_height, header.slot_count)
        != header.total_size
    ):
        raise StageError("Frame transport capacity is inconsistent")
    if header.width <= 0 or header.height <= 0:
        raise StageError("Frame transport active dimensions are invalid")
    if header.width > header.capacity_width or header.height > header.capacity_height:
        raise StageError("Frame transport active dimensions exceed capacity")
    if header.stride != header.width * 3 or header.pixel_format != PixelFormat.RGB24:
        raise StageError("Frame transport pixel layout is unsupported")
    if header.fps_numerator <= 0 or header.fps_denominator <= 0:
        raise StageError("Frame transport frame rate is invalid")
    if header.published_slot >= header.slot_count and header.published_sequence:
        raise StageError("Frame transport published slot is invalid")
    try:
        TransportState(header.state)
    except ValueError as exc:
        raise StageError("Frame transport state is invalid") from exc


def _mutex_names(mapping_name: str, count: int) -> tuple[str, tuple[str, ...]]:
    digest = hashlib.sha256(mapping_name.encode("utf-8")).hexdigest()[:16]
    prefix = f"Local\\FaceLive.Transport.{digest}"
    return f"{prefix}.Header", tuple(f"{prefix}.Slot.{index}" for index in range(count))


if os.name == "nt":
    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _kernel32.CreateMutexW.argtypes = (ctypes.c_void_p, ctypes.c_bool, ctypes.c_wchar_p)
    _kernel32.CreateMutexW.restype = ctypes.c_void_p
    _kernel32.WaitForSingleObject.argtypes = (ctypes.c_void_p, ctypes.c_uint32)
    _kernel32.WaitForSingleObject.restype = ctypes.c_uint32
    _kernel32.ReleaseMutex.argtypes = (ctypes.c_void_p,)
    _kernel32.ReleaseMutex.restype = ctypes.c_bool
    _kernel32.CloseHandle.argtypes = (ctypes.c_void_p,)
    _kernel32.CloseHandle.restype = ctypes.c_bool
    _kernel32.OpenProcess.argtypes = (ctypes.c_uint32, ctypes.c_bool, ctypes.c_uint32)
    _kernel32.OpenProcess.restype = ctypes.c_void_p


class _NamedMutex:
    def __init__(self, name: str) -> None:
        if os.name != "nt":
            raise StageError("Shared frame transport currently requires Windows")
        handle = _kernel32.CreateMutexW(None, False, name)
        if not handle:
            raise StageError(f"Could not create frame transport mutex ({ctypes.get_last_error()})")
        self._handle = handle
        self.name = name
        self._owned = False

    def acquire(self, timeout_ms: int = 0) -> bool:
        if self._handle is None:
            raise StageError("Frame transport mutex is closed")
        result = _kernel32.WaitForSingleObject(self._handle, timeout_ms)
        if result in (_WAIT_OBJECT_0, _WAIT_ABANDONED):
            self._owned = True
            return True
        if result == _WAIT_TIMEOUT:
            return False
        raise StageError(f"Could not wait for frame transport mutex ({ctypes.get_last_error()})")

    def release(self) -> None:
        if self._owned:
            if not _kernel32.ReleaseMutex(self._handle):
                raise StageError(
                    f"Could not release frame transport mutex ({ctypes.get_last_error()})"
                )
            self._owned = False

    def close(self) -> None:
        if self._handle is None:
            return
        self.release()
        _kernel32.CloseHandle(self._handle)
        self._handle = None

    def __enter__(self) -> _NamedMutex:
        if not self.acquire(1000):
            raise StageError("Timed out acquiring frame transport mutex")
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.release()


def _pid_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if pid == os.getpid():
        return True
    if os.name == "nt":
        handle = _kernel32.OpenProcess(_SYNCHRONIZE, False, pid)
        if not handle:
            # Access denial is safer to interpret as a live foreign process.
            return ctypes.get_last_error() != 87
        try:
            return _kernel32.WaitForSingleObject(handle, 0) == _WAIT_TIMEOUT
        finally:
            _kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class SharedMemoryFrameSink:
    """Single-producer, single-consumer bounded shared-memory FrameSink."""

    def __init__(
        self,
        name: str = "facelive_frames_v1",
        *,
        slot_count: int = DEFAULT_SLOT_COUNT,
        capacity_width: int = 1280,
        capacity_height: int = 720,
        consumer_timeout_ms: int = 2_000,
        checksum: bool = True,
        require_consumer: bool = True,
    ) -> None:
        if not name or any(
            character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-"
            for character in name
        ):
            raise ValueError(
                "Transport name must use ASCII letters, digits, dot, dash, or underscore"
            )
        if type(slot_count) is not int or not 2 <= slot_count <= MAX_SLOT_COUNT:
            raise ValueError(f"Transport slot count must be between 2 and {MAX_SLOT_COUNT}")
        if type(capacity_width) is not int or capacity_width <= 0:
            raise ValueError("Transport capacity width must be positive")
        if type(capacity_height) is not int or capacity_height <= 0:
            raise ValueError("Transport capacity height must be positive")
        if type(consumer_timeout_ms) is not int or consumer_timeout_ms <= 0:
            raise ValueError("Transport consumer timeout must be positive")
        if type(checksum) is not bool or type(require_consumer) is not bool:
            raise ValueError("Transport checksum and consumer requirement must be booleans")
        self.name = name
        self.slot_count = slot_count
        self.capacity_width = capacity_width
        self.capacity_height = capacity_height
        self.consumer_timeout_ns = consumer_timeout_ms * 1_000_000
        self.checksum = checksum
        self.require_consumer = require_consumer
        self._lock = threading.RLock()
        self._shm: shared_memory.SharedMemory | None = None
        self._header_mutex: _NamedMutex | None = None
        self._slot_mutexes: tuple[_NamedMutex, ...] = ()
        self._session_id: UUID | None = None
        self._next_sequence = 1
        self._unreported_drops = 0
        self._frame_format: FrameFormat | None = None

    def open(self, frame_format: FrameFormat) -> None:
        if frame_format.width > self.capacity_width or frame_format.height > self.capacity_height:
            raise StageError("Output frame exceeds configured shared-memory capacity")
        expected_size = transport_size(self.capacity_width, self.capacity_height, self.slot_count)
        with self._lock:
            if self._shm is not None:
                raise StageError("Frame transport is already open")
            created = False
            try:
                shm = shared_memory.SharedMemory(name=self.name, create=True, size=expected_size)
                created = True
            except FileExistsError:
                try:
                    shm = shared_memory.SharedMemory(name=self.name, create=False)
                except OSError as exc:
                    raise StageError(f"Could not attach existing frame transport: {exc}") from exc
            if shm.size < expected_size:
                shm.close()
                raise StageError(
                    "Existing frame transport has different capacity; stop its consumer first"
                )
            header_name, slot_names = _mutex_names(self.name, self.slot_count)
            header_mutex = _NamedMutex(header_name)
            slot_mutexes = tuple(_NamedMutex(name) for name in slot_names)
            try:
                if not header_mutex.acquire(1_000):
                    raise StageError("Timed out opening frame transport")
                try:
                    restart_count = 0
                    if not created:
                        header = _read_header(shm.buf)
                        _validate_header(header, shm.size)
                        if (
                            header.capacity_width != self.capacity_width
                            or header.capacity_height != self.capacity_height
                            or header.slot_count != self.slot_count
                        ):
                            raise StageError(
                                "Existing frame transport layout differs from configuration"
                            )
                        if header.state in (
                            TransportState.INITIALIZING,
                            TransportState.RUNNING,
                        ) and _pid_is_alive(header.producer_pid):
                            raise StageError(
                                "Frame transport already has live producer PID "
                                f"{header.producer_pid}"
                            )
                        restart_count = header.restart_count + 1
                    session_id = uuid4()
                    session_high, session_low = _split_uuid(session_id)
                    fps = Fraction(str(frame_format.fps)).limit_denominator(1001)
                    HEADER_STRUCT.pack_into(
                        shm.buf,
                        0,
                        MAGIC,
                        PROTOCOL_VERSION,
                        HEADER_SIZE,
                        ENDIAN_MARKER,
                        expected_size,
                        session_high,
                        session_low,
                        self.capacity_width,
                        self.capacity_height,
                        frame_format.width,
                        frame_format.height,
                        frame_format.width * 3,
                        PixelFormat.RGB24,
                        fps.numerator,
                        fps.denominator,
                        self.slot_count,
                        SLOT_HEADER_SIZE,
                        slot_stride(self.capacity_width, self.capacity_height),
                        os.getpid(),
                        0,
                        TransportState.INITIALIZING,
                        0,
                        time.monotonic_ns(),
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        restart_count,
                    )
                finally:
                    header_mutex.release()
                for index, mutex in enumerate(slot_mutexes):
                    with mutex:
                        start = HEADER_SIZE + index * slot_stride(
                            self.capacity_width, self.capacity_height
                        )
                        shm.buf[start : start + SLOT_HEADER_SIZE] = bytes(SLOT_HEADER_SIZE)
                with header_mutex:
                    struct.pack_into("<I", shm.buf, _OFF_STATE, TransportState.RUNNING)
                    struct.pack_into("<Q", shm.buf, _OFF_PRODUCER_HEARTBEAT, time.monotonic_ns())
            except Exception:
                for mutex in slot_mutexes:
                    mutex.close()
                header_mutex.close()
                shm.close()
                if created:
                    try:
                        shm.unlink()
                    except FileNotFoundError:
                        pass
                raise
            self._shm = shm
            self._header_mutex = header_mutex
            self._slot_mutexes = slot_mutexes
            self._session_id = session_id
            self._next_sequence = 1
            self._unreported_drops = 0
            self._frame_format = frame_format

    def publish(self, frame: VideoFrame) -> bool:
        with self._lock:
            shm = self._shm
            header_mutex = self._header_mutex
            session_id = self._session_id
            frame_format = self._frame_format
            if shm is None or header_mutex is None or session_id is None or frame_format is None:
                raise StageError("Frame transport is not open")
            if frame.frame_id < 0 or frame.frame_id > 0xFFFFFFFFFFFFFFFF:
                raise StageError("Frame ID is outside the transport uint64 range")
            if frame.timestamp_ns < 0 or frame.timestamp_ns > 0xFFFFFFFFFFFFFFFF:
                raise StageError("Frame timestamp is outside the transport uint64 range")
            if frame.rgb.shape != (frame_format.height, frame_format.width, 3):
                raise StageError("Published frame dimensions do not match the transport")
            if frame.rgb.dtype != np.uint8 or not frame.rgb.flags.c_contiguous:
                raise StageError("Published transport frames must be contiguous RGB uint8")

            now = time.monotonic_ns()
            if not header_mutex.acquire(0):
                self._record_drop(now)
                return False
            try:
                header = _read_header(shm.buf)
                session_high, session_low = _split_uuid(session_id)
                if (
                    header.session_high != session_high
                    or header.session_low != session_low
                    or header.producer_pid != os.getpid()
                    or header.state != TransportState.RUNNING
                ):
                    raise StageError("Frame transport ownership changed")
                connected = (
                    header.consumer_pid > 0
                    and header.consumer_session_high == session_high
                    and header.consumer_session_low == session_low
                    and now - header.consumer_heartbeat_ns <= self.consumer_timeout_ns
                    and _pid_is_alive(header.consumer_pid)
                )
                struct.pack_into("<Q", shm.buf, _OFF_PRODUCER_HEARTBEAT, now)
                if self._unreported_drops:
                    struct.pack_into(
                        "<Q",
                        shm.buf,
                        _OFF_DROPPED_COUNT,
                        header.dropped_count + self._unreported_drops,
                    )
                    header = _read_header(shm.buf)
                    self._unreported_drops = 0
                if self.require_consumer and not connected:
                    struct.pack_into("<Q", shm.buf, _OFF_DROPPED_COUNT, header.dropped_count + 1)
                    return False
            finally:
                header_mutex.release()

            sequence = self._next_sequence
            selected_index = -1
            selected_mutex: _NamedMutex | None = None
            for offset in range(self.slot_count):
                index = (sequence - 1 + offset) % self.slot_count
                mutex = self._slot_mutexes[index]
                if mutex.acquire(0):
                    selected_index = index
                    selected_mutex = mutex
                    break
            if selected_mutex is None:
                self._record_drop(now)
                return False
            try:
                payload = memoryview(frame.rgb).cast("B")
                payload_bytes = len(payload)
                checksum = zlib.crc32(payload) if self.checksum else 0
                start = HEADER_SIZE + selected_index * slot_stride(
                    self.capacity_width, self.capacity_height
                )
                session_high, session_low = _split_uuid(session_id)
                SLOT_STRUCT.pack_into(
                    shm.buf,
                    start,
                    sequence,
                    session_high,
                    session_low,
                    frame.timestamp_ns,
                    frame.frame_id,
                    payload_bytes,
                    frame_format.width,
                    frame_format.height,
                    frame_format.width * 3,
                    PixelFormat.RGB24,
                    checksum,
                )
                payload_start = start + SLOT_HEADER_SIZE
                shm.buf[payload_start : payload_start + payload_bytes] = payload
                del payload
                if not header_mutex.acquire(0):
                    self._record_drop(now)
                    return False
                try:
                    header = _read_header(shm.buf)
                    if header.session_id != session_id or header.state != TransportState.RUNNING:
                        raise StageError("Frame transport ownership changed during publication")
                    struct.pack_into("<Q", shm.buf, _OFF_PRODUCER_HEARTBEAT, now)
                    struct.pack_into("<Q", shm.buf, _OFF_PUBLISHED_SEQUENCE, sequence)
                    struct.pack_into("<I", shm.buf, _OFF_PUBLISHED_SLOT, selected_index)
                    struct.pack_into(
                        "<Q", shm.buf, _OFF_PUBLISHED_COUNT, header.published_count + 1
                    )
                    if self._unreported_drops:
                        struct.pack_into(
                            "<Q",
                            shm.buf,
                            _OFF_DROPPED_COUNT,
                            header.dropped_count + self._unreported_drops,
                        )
                        self._unreported_drops = 0
                    self._next_sequence += 1
                    return True
                finally:
                    header_mutex.release()
            finally:
                selected_mutex.release()

    def _record_drop(self, now: int) -> None:
        self._unreported_drops += 1
        shm = self._shm
        mutex = self._header_mutex
        if shm is None or mutex is None or not mutex.acquire(0):
            return
        try:
            header = _read_header(shm.buf)
            struct.pack_into("<Q", shm.buf, _OFF_PRODUCER_HEARTBEAT, now)
            struct.pack_into(
                "<Q",
                shm.buf,
                _OFF_DROPPED_COUNT,
                header.dropped_count + self._unreported_drops,
            )
            self._unreported_drops = 0
        finally:
            mutex.release()

    def status(self) -> TransportStatus:
        with self._lock:
            if self._shm is None or self._header_mutex is None:
                raise StageError("Frame transport is not open")
            with self._header_mutex:
                header = _read_header(self._shm.buf)
                dropped_count = header.dropped_count + self._unreported_drops
                if self._unreported_drops:
                    struct.pack_into("<Q", self._shm.buf, _OFF_DROPPED_COUNT, dropped_count)
                    self._unreported_drops = 0
                    header = _read_header(self._shm.buf)
            return _status_from_header(header, self.consumer_timeout_ns)

    def close(self) -> None:
        with self._lock:
            shm, self._shm = self._shm, None
            header_mutex, self._header_mutex = self._header_mutex, None
            slot_mutexes, self._slot_mutexes = self._slot_mutexes, ()
            session_id, self._session_id = self._session_id, None
            self._frame_format = None
            self._unreported_drops = 0
            try:
                if shm is not None and header_mutex is not None and session_id is not None:
                    if header_mutex.acquire(1_000):
                        try:
                            header = _read_header(shm.buf)
                            if (
                                header.session_id == session_id
                                and header.producer_pid == os.getpid()
                            ):
                                struct.pack_into("<I", shm.buf, _OFF_STATE, TransportState.STOPPED)
                                struct.pack_into("<I", shm.buf, _OFF_PRODUCER_PID, 0)
                                struct.pack_into(
                                    "<Q",
                                    shm.buf,
                                    _OFF_PRODUCER_HEARTBEAT,
                                    time.monotonic_ns(),
                                )
                        except (StageError, ValueError):
                            # A replaced or corrupt mapping must not prevent handle cleanup.
                            pass
                        finally:
                            header_mutex.release()
            finally:
                for mutex in slot_mutexes:
                    mutex.close()
                if header_mutex is not None:
                    header_mutex.close()
                if shm is not None:
                    shm.close()


class SharedMemoryFrameConsumer:
    """Standalone newest-frame consumer for the versioned shared-memory ABI."""

    def __init__(
        self,
        name: str = "facelive_frames_v1",
        *,
        producer_timeout_ms: int = 2_000,
        verify_checksum: bool = True,
    ) -> None:
        if not name:
            raise ValueError("Transport name is required")
        if type(producer_timeout_ms) is not int or producer_timeout_ms <= 0:
            raise ValueError("Producer timeout must be positive")
        self.name = name
        self.producer_timeout_ns = producer_timeout_ms * 1_000_000
        self.verify_checksum = verify_checksum
        self._shm: shared_memory.SharedMemory | None = None
        self._header_mutex: _NamedMutex | None = None
        self._slot_mutexes: tuple[_NamedMutex, ...] = ()
        self._last_session: UUID | None = None
        self._last_sequence = 0
        self.skipped_frames = 0

    def open(self) -> None:
        if self._shm is not None:
            raise StageError("Frame transport consumer is already open")
        try:
            shm = shared_memory.SharedMemory(name=self.name, create=False)
        except FileNotFoundError as exc:
            raise StageError("Frame transport producer is not available") from exc
        header_name, _ = _mutex_names(self.name, DEFAULT_SLOT_COUNT)
        header_mutex = _NamedMutex(header_name)
        try:
            with header_mutex:
                header = _read_header(shm.buf)
                _validate_header(header, shm.size)
                if (
                    header.consumer_pid
                    and header.consumer_session_high == header.session_high
                    and header.consumer_session_low == header.session_low
                    and _pid_is_alive(header.consumer_pid)
                    and time.monotonic_ns() - header.consumer_heartbeat_ns
                    <= self.producer_timeout_ns
                ):
                    raise StageError(
                        f"Frame transport already has live consumer PID {header.consumer_pid}"
                    )
                struct.pack_into("<I", shm.buf, _OFF_CONSUMER_PID, os.getpid())
                struct.pack_into("<Q", shm.buf, _OFF_CONSUMER_HEARTBEAT, time.monotonic_ns())
                struct.pack_into("<Q", shm.buf, _OFF_CONSUMER_SESSION_HIGH, header.session_high)
                struct.pack_into("<Q", shm.buf, _OFF_CONSUMER_SESSION_LOW, header.session_low)
            _, slot_names = _mutex_names(self.name, header.slot_count)
            slot_mutexes = tuple(_NamedMutex(name) for name in slot_names)
        except Exception:
            header_mutex.close()
            shm.close()
            raise
        self._shm = shm
        self._header_mutex = header_mutex
        self._slot_mutexes = slot_mutexes
        self._last_session = None
        self._last_sequence = 0
        self.skipped_frames = 0

    def read_latest(self) -> TransportFrame | None:
        shm = self._shm
        header_mutex = self._header_mutex
        if shm is None or header_mutex is None:
            raise StageError("Frame transport consumer is not open")
        if not header_mutex.acquire(0):
            return None
        try:
            header = _read_header(shm.buf)
            _validate_header(header, shm.size)
            now = time.monotonic_ns()
            struct.pack_into("<I", shm.buf, _OFF_CONSUMER_PID, os.getpid())
            struct.pack_into("<Q", shm.buf, _OFF_CONSUMER_HEARTBEAT, now)
            struct.pack_into("<Q", shm.buf, _OFF_CONSUMER_SESSION_HIGH, header.session_high)
            struct.pack_into("<Q", shm.buf, _OFF_CONSUMER_SESSION_LOW, header.session_low)
            if header.state != TransportState.RUNNING or header.published_sequence == 0:
                return None
            session_id = header.session_id
            sequence = header.published_sequence
            slot_index = header.published_slot
        finally:
            header_mutex.release()
        if session_id == self._last_session and sequence <= self._last_sequence:
            return None
        slot_mutex = self._slot_mutexes[slot_index]
        if not slot_mutex.acquire(0):
            return None
        try:
            start = HEADER_SIZE + slot_index * header.slot_stride
            (
                slot_sequence,
                session_high,
                session_low,
                timestamp_ns,
                frame_id,
                payload_bytes,
                width,
                height,
                stride,
                pixel_format,
                checksum,
            ) = SLOT_STRUCT.unpack_from(shm.buf, start)
            if (
                slot_sequence != sequence
                or session_high != header.session_high
                or session_low != header.session_low
            ):
                return None
            if pixel_format != PixelFormat.RGB24 or stride != width * 3:
                raise StageError("Shared frame slot uses an unsupported pixel layout")
            if width != header.width or height != header.height:
                raise StageError("Shared frame slot dimensions do not match the session")
            if (
                payload_bytes != height * stride
                or payload_bytes > header.slot_stride - SLOT_HEADER_SIZE
            ):
                raise StageError("Shared frame payload length is invalid")
            payload_start = start + SLOT_HEADER_SIZE
            payload = bytes(shm.buf[payload_start : payload_start + payload_bytes])
            if self.verify_checksum and checksum and zlib.crc32(payload) != checksum:
                raise StageError("Shared frame checksum mismatch")
        finally:
            slot_mutex.release()
        pixels = np.frombuffer(payload, dtype=np.uint8).reshape(height, width, 3)
        pixels.setflags(write=False)
        if self._last_session == session_id and sequence > self._last_sequence + 1:
            self.skipped_frames += sequence - self._last_sequence - 1
        self._last_session = session_id
        self._last_sequence = sequence
        return TransportFrame(
            session_id,
            sequence,
            frame_id,
            timestamp_ns,
            width,
            height,
            stride,
            PixelFormat(pixel_format),
            pixels,
        )

    def status(self) -> TransportStatus:
        if self._shm is None or self._header_mutex is None:
            raise StageError("Frame transport consumer is not open")
        with self._header_mutex:
            header = _read_header(self._shm.buf)
            _validate_header(header, self._shm.size)
        return _status_from_header(header, self.producer_timeout_ns)

    def close(self) -> None:
        shm, self._shm = self._shm, None
        header_mutex, self._header_mutex = self._header_mutex, None
        slot_mutexes, self._slot_mutexes = self._slot_mutexes, ()
        try:
            if shm is not None and header_mutex is not None and header_mutex.acquire(100):
                try:
                    header = _read_header(shm.buf)
                    if header.consumer_pid == os.getpid():
                        struct.pack_into("<I", shm.buf, _OFF_CONSUMER_PID, 0)
                        struct.pack_into("<Q", shm.buf, _OFF_CONSUMER_HEARTBEAT, 0)
                        struct.pack_into("<Q", shm.buf, _OFF_CONSUMER_SESSION_HIGH, 0)
                        struct.pack_into("<Q", shm.buf, _OFF_CONSUMER_SESSION_LOW, 0)
                except (StageError, ValueError):
                    # Detach even if another process damaged or replaced the header.
                    pass
                finally:
                    header_mutex.release()
        finally:
            for mutex in slot_mutexes:
                mutex.close()
            if header_mutex is not None:
                header_mutex.close()
            if shm is not None:
                shm.close()
            self._last_session = None
            self._last_sequence = 0


def _status_from_header(header: _Header, peer_timeout_ns: int) -> TransportStatus:
    now = time.monotonic_ns()
    producer_alive = (
        header.state == TransportState.RUNNING
        and header.producer_pid > 0
        and now - header.producer_heartbeat_ns <= peer_timeout_ns
        and _pid_is_alive(header.producer_pid)
    )
    consumer_connected = (
        header.consumer_pid > 0
        and header.consumer_session_high == header.session_high
        and header.consumer_session_low == header.session_low
        and now - header.consumer_heartbeat_ns <= peer_timeout_ns
        and _pid_is_alive(header.consumer_pid)
    )
    return TransportStatus(
        session_id=header.session_id,
        state=TransportState(header.state),
        producer_pid=header.producer_pid,
        consumer_pid=header.consumer_pid,
        producer_heartbeat_ns=header.producer_heartbeat_ns,
        consumer_heartbeat_ns=header.consumer_heartbeat_ns,
        published_sequence=header.published_sequence,
        published_count=header.published_count,
        dropped_count=header.dropped_count,
        restart_count=header.restart_count,
        producer_alive=producer_alive,
        consumer_connected=consumer_connected,
        error_code=header.error_code,
    )
