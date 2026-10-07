"""Benchmark tracking and deterministic processing on the controlled motion clip."""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import cv2
import numpy as np

from app.compositing import AlphaFaceCompositor
from app.pipeline.types import FrameFormat, TrackingStatus, VideoFrame
from app.reference import PoseSpaceReferenceSelector, ReferenceLibraryStore
from app.rendering import GeometricFaceRenderer
from app.stabilization import TemporalStabilizer
from app.tracking import MediaPipeFaceTracker


def _summary(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    return statistics.fmean(values), float(np.percentile(values, 95))


def main() -> int:
    repository = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--clip", type=Path, default=repository / "tests" / "assets" / "controlled_motion.mp4"
    )
    parser.add_argument("--references", type=Path, default=repository / "assets" / "pose_guides")
    parser.add_argument("--max-frames", type=int, default=240)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--warmup-frames", type=int, default=15)
    parser.add_argument("--disable-smoothing", action="store_true")
    args = parser.parse_args()
    if args.max_frames <= args.warmup_frames or args.warmup_frames < 0 or args.start_frame < 0:
        parser.error("--max-frames must be greater than nonnegative --warmup-frames")

    capture = cv2.VideoCapture(str(args.clip))
    if not capture.isOpened():
        print(f"Could not open clip: {args.clip}")
        return 2
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if width <= 0 or height <= 0 or fps <= 0:
        print("Clip has invalid dimensions or FPS")
        capture.release()
        return 2
    capture.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)

    library = ReferenceLibraryStore()
    tracker = MediaPipeFaceTracker()
    renderer = GeometricFaceRenderer()
    compositor = AlphaFaceCompositor()
    selector = PoseSpaceReferenceSelector()
    stabilizer = TemporalStabilizer()
    try:
        for slot in library.slots():
            library.add_reference(slot.slot_id, args.references / f"{slot.slot_id}.png")
        references = library.references()
        frame_format = FrameFormat(width, height, fps)
        tracker.open()
        renderer.open(frame_format)

        tracking_latencies: list[float] = []
        processing_latencies: list[float] = []
        total_latencies: list[float] = []
        measured_frames = 0
        processed_frames = 0
        no_face_frames = 0
        held_frames = 0
        benchmark_started = 0.0
        frame_period_ns = round(1_000_000_000 / fps)

        for offset in range(args.max_frames):
            frame_id = args.start_frame + offset
            ok, bgr = capture.read()
            if not ok:
                break
            if offset == args.warmup_frames:
                benchmark_started = time.perf_counter()
            rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            rgb.setflags(write=False)
            timestamp_ns = 1_000_000_000 + frame_id * frame_period_ns
            frame = VideoFrame(frame_id, timestamp_ns, rgb)
            tracking_started = time.perf_counter_ns()
            if not tracker.submit(frame):
                raise RuntimeError(
                    "Tracker unexpectedly rejected a frame with no request in flight"
                )
            deadline = time.monotonic() + 2.0
            result = None
            while result is None and time.monotonic() < deadline:
                result = tracker.poll_latest()
                if result is None:
                    time.sleep(0.0005)
            if result is None:
                raise RuntimeError(f"Tracker timed out on frame {frame_id}")

            face = result.face
            held = False
            processing_started = time.perf_counter_ns()
            if args.disable_smoothing:
                pass
            elif result.status is TrackingStatus.TRACKED:
                face = stabilizer.update(result.face)
            elif result.status is TrackingStatus.NO_FACE:
                face = stabilizer.coast(result.frame_id, result.timestamp_ns)
                held = face is not None
            if face is not None:
                weights = selector.select(face, references)
                if not args.disable_smoothing:
                    weights = stabilizer.smooth_weights(weights, face.timestamp_ns)
                rendered = renderer.render(face, references, weights)
                compositor.composite(frame, rendered)
            processing_finished = time.perf_counter_ns()

            if offset < args.warmup_frames:
                continue
            measured_frames += 1
            if face is None:
                no_face_frames += 1
            else:
                processed_frames += 1
                held_frames += int(held)
                processing_latencies.append((processing_finished - processing_started) / 1_000_000)
            if result.latency_ms is not None:
                tracking_latencies.append(result.latency_ms)
            total_latencies.append((processing_finished - tracking_started) / 1_000_000)

        elapsed = time.perf_counter() - benchmark_started
        tracking_mean, tracking_p95 = _summary(tracking_latencies)
        processing_mean, processing_p95 = _summary(processing_latencies)
        total_mean, total_p95 = _summary(total_latencies)
        print(f"Clip: {args.clip}")
        print(f"Mode: {'raw' if args.disable_smoothing else 'temporally smoothed'}")
        print(f"Measured frames: {measured_frames}; processed: {processed_frames}")
        print(f"No-face frames: {no_face_frames}; held frames: {held_frames}")
        print(f"Pipeline throughput: {measured_frames / elapsed:.2f} FPS")
        print(f"Processed output throughput: {processed_frames / elapsed:.2f} FPS")
        print(f"Tracking latency: mean {tracking_mean:.2f} ms; p95 {tracking_p95:.2f} ms")
        print(f"Processing latency: mean {processing_mean:.2f} ms; p95 {processing_p95:.2f} ms")
        print(f"Total latency: mean {total_mean:.2f} ms; p95 {total_p95:.2f} ms")
        return 0
    finally:
        capture.release()
        renderer.close()
        tracker.close()
        library.close()


if __name__ == "__main__":
    raise SystemExit(main())
