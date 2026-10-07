"""Windows webcam adapter backed by OpenCV.

OpenCV types and BGR images are confined to this module. Published frames follow the
backend-independent RGB contract from :mod:`app.pipeline.types`.
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from cv2_enumerate_cameras import enumerate_cameras

from app.pipeline.types import CameraDevice, FrameFormat, StageError, VideoFrame

_VIRTUAL_CAMERA_MARKERS = (
    "virtual",
    "obs camera",
    "obs-camera",
    "manycam",
    "snap camera",
    "xsplit",
    "ndi video",
    "camo camera",
)


@dataclass(frozen=True, slots=True)
class _DeviceSpec:
    public: CameraDevice
    index: int
    backend: int


def _is_virtual_camera(name: str) -> bool:
    normalized = name.casefold()
    return any(marker in normalized for marker in _VIRTUAL_CAMERA_MARKERS)


def _device_id(path: str, backend: int, index: int) -> str:
    identity = path.casefold() if path else f"index:{index}"
    digest = hashlib.sha256(f"{backend}:{identity}".encode()).hexdigest()[:20]
    return f"opencv:{digest}"


class OpenCVCameraSource:
    """A latest-frame camera source with a dedicated blocking capture thread."""

    def __init__(
        self,
        *,
        disconnect_timeout_seconds: float = 2.0,
        enumerator: Callable[[int], Iterable[Any]] = enumerate_cameras,
        capture_factory: Callable[[int, int], Any] = cv2.VideoCapture,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        if disconnect_timeout_seconds <= 0:
            raise ValueError("disconnect_timeout_seconds must be positive")
        self._disconnect_timeout_seconds = disconnect_timeout_seconds
        self._enumerator = enumerator
        self._capture_factory = capture_factory
        self._monotonic_ns = monotonic_ns
        self._devices: dict[str, _DeviceSpec] = {}
        self._capture: Any | None = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._state_lock = threading.Lock()
        self._latest: VideoFrame | None = None
        self._last_delivered_id = -1
        self._next_frame_id = 0
        self._last_capture_timestamp_ns = -1
        self._error: StageError | None = None
        self._logger = logging.getLogger("facelive.camera")

    def enumerate_devices(self) -> tuple[CameraDevice, ...]:
        """Enumerate named Windows video inputs without opening them."""
        discovered: list[Any] = []
        errors: list[Exception] = []
        for backend in (cv2.CAP_MSMF, cv2.CAP_DSHOW):
            try:
                discovered = list(self._enumerator(backend))
            except Exception as exc:  # The package wraps native platform enumeration.
                errors.append(exc)
                self._logger.warning("Camera enumeration failed for backend %s: %s", backend, exc)
            if discovered:
                break
        if not discovered and errors:
            raise StageError(f"Camera enumeration failed: {errors[-1]}") from errors[-1]

        specs: dict[str, _DeviceSpec] = {}
        for item in discovered:
            name = str(item.name).strip() or f"Camera {item.index + 1}"
            public = CameraDevice(
                device_id=_device_id(str(item.path), int(item.backend), int(item.index)),
                display_name=name,
                is_virtual=_is_virtual_camera(name),
            )
            specs[public.device_id] = _DeviceSpec(public, int(item.index), int(item.backend))
        self._devices = specs
        return tuple(spec.public for spec in specs.values())

    def open(self, device_id: str, requested: FrameFormat) -> FrameFormat:
        """Open and validate a camera, then begin retaining only its newest frame."""
        self.close()
        spec = self._devices.get(device_id)
        if spec is None:
            self.enumerate_devices()
            spec = self._devices.get(device_id)
        if spec is None:
            raise StageError("The selected camera is no longer available")
        if spec.public.is_virtual:
            raise StageError("Select a physical camera, not a virtual camera")

        capture = self._capture_factory(spec.index, spec.backend)
        if capture is None or not capture.isOpened():
            if capture is not None:
                capture.release()
            raise StageError(f"Could not open {spec.public.display_name}")

        try:
            # MJPG commonly permits 720p/30 over USB. Every property is a request; the
            # values after the first frame define the actual negotiated format.
            capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, requested.width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, requested.height)
            capture.set(cv2.CAP_PROP_FPS, requested.fps)
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            ok, bgr = capture.read()
            if not ok or bgr is None:
                raise StageError(f"{spec.public.display_name} opened but returned no video")
            frame = self._make_frame(bgr, 0)
            height, width = frame.rgb.shape[:2]
            reported_fps = float(capture.get(cv2.CAP_PROP_FPS))
            negotiated = FrameFormat(
                width, height, reported_fps if reported_fps > 0 else requested.fps
            )
        except Exception as exc:
            capture.release()
            if isinstance(exc, StageError):
                raise
            raise StageError(f"Could not initialize {spec.public.display_name}: {exc}") from exc

        with self._state_lock:
            self._capture = capture
            self._latest = frame
            self._last_delivered_id = -1
            self._next_frame_id = 1
            self._error = None
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._capture_loop,
            args=(capture, spec.public.display_name),
            name="facelive-camera-capture",
            daemon=True,
        )
        self._thread.start()
        self._logger.info(
            "Opened %s at %dx%d, requested %.1f FPS, backend %s",
            spec.public.display_name,
            negotiated.width,
            negotiated.height,
            requested.fps,
            spec.backend,
        )
        return negotiated

    def _make_frame(self, bgr: Any, frame_id: int) -> VideoFrame:
        if not isinstance(bgr, np.ndarray) or bgr.ndim != 3 or bgr.shape[2] != 3:
            raise StageError("Camera returned an unsupported image format")
        rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), dtype=np.uint8)
        rgb.setflags(write=False)
        timestamp_ns = max(self._monotonic_ns(), self._last_capture_timestamp_ns + 1)
        self._last_capture_timestamp_ns = timestamp_ns
        return VideoFrame(frame_id=frame_id, timestamp_ns=timestamp_ns, rgb=rgb)

    def _capture_loop(self, capture: Any, display_name: str) -> None:
        failed_since: float | None = None
        while not self._stop_event.is_set():
            ok, bgr = capture.read()
            if ok and bgr is not None:
                failed_since = None
                with self._state_lock:
                    frame_id = self._next_frame_id
                    self._next_frame_id += 1
                try:
                    frame = self._make_frame(bgr, frame_id)
                except StageError as exc:
                    with self._state_lock:
                        self._error = exc
                    break
                with self._state_lock:
                    self._latest = frame
                continue

            now = time.monotonic()
            failed_since = failed_since or now
            if now - failed_since >= self._disconnect_timeout_seconds:
                with self._state_lock:
                    self._error = StageError(f"Lost video from {display_name}")
                break
            self._stop_event.wait(0.05)

    def read_latest(self) -> VideoFrame | None:
        with self._state_lock:
            if self._error is not None:
                raise self._error
            frame = self._latest
            if frame is None or frame.frame_id == self._last_delivered_id:
                return None
            self._last_delivered_id = frame.frame_id
            return frame

    def close(self) -> None:
        """Stop capture and release the camera. Safe to call repeatedly."""
        self._stop_event.set()
        thread = self._thread
        capture = self._capture
        if thread is not None and thread.is_alive():
            thread.join(timeout=0.5)
        if capture is not None:
            capture.release()
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.5)
        if thread is not None and thread.is_alive():
            self._logger.warning("Camera capture worker did not stop within two seconds")
        with self._state_lock:
            self._capture = None
            self._thread = None
            self._latest = None
            self._last_delivered_id = -1
            self._next_frame_id = 0
            self._last_capture_timestamp_ns = -1
            self._error = None
