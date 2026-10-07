"""Benchmark the bounded, overlapped 1280x720 processed-preview schedule."""

from __future__ import annotations

import argparse
import os
import statistics
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap, QTransform
from PySide6.QtWidgets import QApplication, QLabel

from app.compositing import AlphaFaceCompositor
from app.pipeline.types import FaceState, FrameFormat, VideoFrame
from app.reference import PoseSpaceReferenceSelector, ReferenceLibraryStore
from app.rendering import GeometricFaceRenderer
from app.stabilization import TemporalStabilizer
from app.tracking import MediaPipeFaceTracker


@dataclass(frozen=True, slots=True)
class ProcessingResult:
    output: VideoFrame
    rendering_ms: float
    compositing_ms: float
    captured_at_ns: int
    measured: bool


def summary(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    return statistics.fmean(values), float(np.percentile(values, 95))


def print_stage(name: str, values: list[float]) -> None:
    mean, p95 = summary(values)
    print(f"{name:<22} mean {mean:7.2f} ms; p95 {p95:7.2f} ms")


def process_frame(
    renderer,
    compositor,
    frame,
    face,
    references,
    weights,
    captured_at_ns,
    measured,
) -> ProcessingResult:
    started = time.perf_counter_ns()
    rendered = renderer.render(face, references, weights)
    rendered_at = time.perf_counter_ns()
    output = compositor.composite(frame, rendered)
    completed_at = time.perf_counter_ns()
    return ProcessingResult(
        output,
        (rendered_at - started) / 1_000_000,
        (completed_at - rendered_at) / 1_000_000,
        captured_at_ns,
        measured,
    )


def present(label: QLabel, app: QApplication, frame: VideoFrame) -> tuple[float, float]:
    started = time.perf_counter_ns()
    height, width = frame.rgb.shape[:2]
    image = QImage(
        frame.rgb.data,
        width,
        height,
        int(frame.rgb.strides[0]),
        QImage.Format.Format_RGB888,
    ).copy()
    copied_at = time.perf_counter_ns()
    display_image = image.scaled(
        label.size(),
        Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.FastTransformation,
    ).transformed(
        QTransform().scale(-1.0, 1.0),
        Qt.TransformationMode.FastTransformation,
    )
    label.setPixmap(QPixmap.fromImage(display_image))
    app.processEvents()
    finished = time.perf_counter_ns()
    return (copied_at - started) / 1_000_000, (finished - copied_at) / 1_000_000


def main() -> int:
    repository = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--clip", type=Path, default=repository / "tests" / "assets" / "controlled_motion.mp4"
    )
    parser.add_argument("--references", type=Path, default=repository / "assets" / "pose_guides")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--tracking-fps", type=float, default=10.0)
    parser.add_argument("--working-resolution", type=int, default=192)
    parser.add_argument("--opencv-threads", type=int, default=4)
    parser.add_argument("--save-frame", type=Path)
    parser.add_argument("--frames", type=int, default=150)
    parser.add_argument("--warmup-frames", type=int, default=30)
    args = parser.parse_args()
    if (
        args.width <= 0
        or args.height <= 0
        or args.fps <= 0
        or args.tracking_fps <= 0
        or args.tracking_fps > args.fps
        or args.frames <= args.warmup_frames
        or args.warmup_frames < 0
    ):
        parser.error(
            "dimensions/FPS must be positive, tracking FPS cannot exceed input FPS, "
            "and frames must exceed warmup frames"
        )
    if args.opencv_threads < 0:
        parser.error("OpenCV thread count must be nonnegative")
    if args.opencv_threads:
        cv2.setNumThreads(args.opencv_threads)

    capture = cv2.VideoCapture(str(args.clip))
    if not capture.isOpened():
        print(f"Could not open clip: {args.clip}")
        return 2
    app = QApplication.instance() or QApplication(["facelive-realtime-benchmark"])
    label = QLabel()
    label.resize(960, 540)
    library = ReferenceLibraryStore()
    tracker = MediaPipeFaceTracker()
    renderer = GeometricFaceRenderer(working_resolution=args.working_resolution)
    compositor = AlphaFaceCompositor()
    selector = PoseSpaceReferenceSelector()
    stabilizer = TemporalStabilizer()
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="benchmark-processing")
    pending_tracking: tuple[VideoFrame, int, bool] | None = None
    processing: Future[ProcessingResult] | None = None
    pending_processing: tuple[VideoFrame, FaceState, tuple, int, bool] | None = None
    latest_face: FaceState | None = None
    latest_weights = ()
    tracking_drops = 0
    tracking_throttled = 0
    processing_drops = 0
    outputs = 0
    last_output: VideoFrame | None = None
    stages: dict[str, list[float]] = {
        name: []
        for name in (
            "capture_convert",
            "tracking",
            "stabilization",
            "reference_selection",
            "rendering",
            "compositing",
            "ui_copy",
            "ui_presentation",
            "full_frame",
        )
    }
    measurement_started_ns = 0
    frame_period = 1.0 / args.fps
    tracking_stride = max(1, round(args.fps / args.tracking_fps))
    next_frame_at = time.perf_counter()
    sent = 0
    shutdown_ms = 0.0

    try:
        for slot in library.slots():
            library.add_reference(slot.slot_id, args.references / f"{slot.slot_id}.png")
        references = library.references()
        tracker.open()
        renderer.open(FrameFormat(args.width, args.height, args.fps))

        while (
            sent < args.frames
            or pending_tracking is not None
            or processing is not None
            or pending_processing is not None
        ):
            if processing is not None and processing.done():
                result = processing.result()
                processing = None
                last_output = result.output
                ui_copy_ms, ui_presentation_ms = present(label, app, result.output)
                if result.measured:
                    presented_at_ns = time.perf_counter_ns()
                    outputs += 1
                    stages["rendering"].append(result.rendering_ms)
                    stages["compositing"].append(result.compositing_ms)
                    stages["ui_copy"].append(ui_copy_ms)
                    stages["ui_presentation"].append(ui_presentation_ms)
                    stages["full_frame"].append(
                        (presented_at_ns - result.captured_at_ns) / 1_000_000
                    )
                if pending_processing is not None:
                    (
                        queued_frame,
                        queued_face,
                        queued_weights,
                        queued_at_ns,
                        queued_measured,
                    ) = pending_processing
                    pending_processing = None
                    processing = executor.submit(
                        process_frame,
                        renderer,
                        compositor,
                        queued_frame,
                        queued_face,
                        references,
                        queued_weights,
                        queued_at_ns,
                        queued_measured,
                    )

            result = tracker.poll_latest()
            if result is not None:
                retained, pending_tracking = pending_tracking, None
                if retained is None or (
                    retained[0].frame_id,
                    retained[0].timestamp_ns,
                ) != (result.frame_id, result.timestamp_ns):
                    raise RuntimeError("Tracking result lost its matching retained frame")
                frame, full_frame_started_ns, measured = retained
                had_face = latest_face is not None
                smoothing_started = time.perf_counter_ns()
                face: FaceState | None
                if result.face is not None:
                    face = stabilizer.update(result.face)
                else:
                    face = stabilizer.coast(result.frame_id, result.timestamp_ns)
                smoothing_finished = time.perf_counter_ns()
                selection_started = time.perf_counter_ns()
                weights = selector.select(face, references) if face is not None else ()
                if face is not None:
                    weights = stabilizer.smooth_weights(weights, face.timestamp_ns)
                    latest_face = face
                    latest_weights = weights
                selection_finished = time.perf_counter_ns()
                if measured:
                    if result.latency_ms is not None:
                        stages["tracking"].append(result.latency_ms)
                    stages["stabilization"].append(
                        (smoothing_finished - smoothing_started) / 1_000_000
                    )
                    stages["reference_selection"].append(
                        (selection_finished - selection_started) / 1_000_000
                    )
                if not had_face and face is not None and weights:
                    if processing is None:
                        processing = executor.submit(
                            process_frame,
                            renderer,
                            compositor,
                            frame,
                            face,
                            references,
                            weights,
                            full_frame_started_ns,
                            measured,
                        )
                    else:
                        if pending_processing is not None and measured:
                            processing_drops += 1
                        pending_processing = (
                            frame,
                            face,
                            weights,
                            full_frame_started_ns,
                            measured,
                        )

            now = time.perf_counter()
            if sent < args.frames and now >= next_frame_at:
                capture_started = time.perf_counter_ns()
                ok, bgr = capture.read()
                if not ok:
                    raise RuntimeError(f"Clip ended at frame {sent}")
                if bgr.shape[1] != args.width or bgr.shape[0] != args.height:
                    bgr = cv2.resize(bgr, (args.width, args.height), interpolation=cv2.INTER_LINEAR)
                rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), dtype=np.uint8)
                rgb.setflags(write=False)
                captured_at_ns = time.perf_counter_ns()
                full_frame_started_ns = capture_started
                measured = sent >= args.warmup_frames
                if sent == args.warmup_frames:
                    measurement_started_ns = capture_started
                if measured:
                    stages["capture_convert"].append((captured_at_ns - capture_started) / 1_000_000)
                frame = VideoFrame(sent, captured_at_ns, rgb)
                if sent % tracking_stride:
                    if measured:
                        tracking_throttled += 1
                elif tracker.submit(frame):
                    pending_tracking = (frame, full_frame_started_ns, measured)
                elif measured:
                    tracking_drops += 1
                if latest_face is not None and latest_weights:
                    current_face = replace(
                        latest_face,
                        frame_id=frame.frame_id,
                        timestamp_ns=frame.timestamp_ns,
                    )
                    if processing is None:
                        processing = executor.submit(
                            process_frame,
                            renderer,
                            compositor,
                            frame,
                            current_face,
                            references,
                            latest_weights,
                            full_frame_started_ns,
                            measured,
                        )
                    else:
                        if pending_processing is not None and measured:
                            processing_drops += 1
                        pending_processing = (
                            frame,
                            current_face,
                            latest_weights,
                            full_frame_started_ns,
                            measured,
                        )
                sent += 1
                next_frame_at += frame_period
                if next_frame_at < now - frame_period:
                    next_frame_at = now + frame_period

            app.processEvents()
            time.sleep(0.0005)

        finished_ns = time.perf_counter_ns()
        elapsed = (finished_ns - measurement_started_ns) / 1_000_000_000
        print(f"Input: {args.clip}")
        print(f"Format: {args.width}x{args.height} at {args.fps:.1f} FPS")
        print(f"Measured input frames: {args.frames - args.warmup_frames}")
        print(f"Processed outputs: {outputs}; output throughput: {outputs / elapsed:.2f} FPS")
        print(
            f"Tracker drops: {tracking_drops}; tracker throttled: {tracking_throttled}; "
            f"processing drops: {processing_drops}"
        )
        print(
            "Maximum queue depth: tracker 1; processing active 1 + latest pending 1; presentation 1"
        )
        for name, values in stages.items():
            print_stage(name, values)
        print("Neural inference       n/a (geometric backend selected)")
        if args.save_frame is not None and last_output is not None:
            args.save_frame.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(
                str(args.save_frame), cv2.cvtColor(last_output.rgb, cv2.COLOR_RGB2BGR)
            ):
                raise RuntimeError(f"Could not save benchmark frame: {args.save_frame}")
            print(f"Saved output frame: {args.save_frame}")
        return 0
    finally:
        shutdown_started = time.perf_counter_ns()
        if processing is not None:
            processing.cancel()
        pending_processing = None
        executor.shutdown(wait=True, cancel_futures=True)
        capture.release()
        renderer.close()
        tracker.close()
        library.close()
        label.clear()
        shutdown_ms = (time.perf_counter_ns() - shutdown_started) / 1_000_000
        print(f"Clean resource shutdown: {shutdown_ms:.2f} ms")


if __name__ == "__main__":
    raise SystemExit(main())
