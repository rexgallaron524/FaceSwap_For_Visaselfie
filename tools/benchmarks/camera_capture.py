"""Measure physical-camera capture independently of the Qt preview."""

from __future__ import annotations

import argparse
import time

from app.camera import OpenCVCameraSource
from app.pipeline.types import FrameFormat, StageError


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--device", type=int, default=0, help="Physical-camera list index")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=float, default=30)
    args = parser.parse_args()
    if args.seconds <= 0:
        parser.error("--seconds must be positive")

    source = OpenCVCameraSource()
    try:
        devices = tuple(device for device in source.enumerate_devices() if not device.is_virtual)
        if not devices:
            print("No physical camera found")
            return 2
        if args.device < 0 or args.device >= len(devices):
            camera_count = len(devices)
            print(
                f"Camera index {args.device} is unavailable; "
                f"found {camera_count} physical camera(s)"
            )
            return 2
        device = devices[args.device]
        negotiated = source.open(device.device_id, FrameFormat(args.width, args.height, args.fps))
        start = time.monotonic()
        deadline = start + args.seconds
        first = None
        last = None
        delivered = 0
        skipped = 0
        while time.monotonic() < deadline:
            frame = source.read_latest()
            if frame is None:
                time.sleep(0.001)
                continue
            if last is not None:
                skipped += max(0, frame.frame_id - last.frame_id - 1)
            first = first or frame
            last = frame
            delivered += 1
        elapsed = time.monotonic() - start
        capture_fps = 0.0
        if first is not None and last is not None and last.timestamp_ns > first.timestamp_ns:
            capture_fps = (last.frame_id - first.frame_id) / (
                (last.timestamp_ns - first.timestamp_ns) / 1_000_000_000
            )
        print(f"Device: {device.display_name}")
        print(
            f"Negotiated: {negotiated.width}x{negotiated.height} at "
            f"{negotiated.fps:.1f} reported FPS"
        )
        print(f"Observed capture FPS: {capture_fps:.2f}")
        print(f"Delivered polling FPS: {delivered / elapsed:.2f}")
        print(f"Preview-poll skipped frames: {skipped}")
        return 0
    except StageError as exc:
        print(f"Camera benchmark failed: {exc}")
        return 1
    finally:
        source.close()


if __name__ == "__main__":
    raise SystemExit(main())
