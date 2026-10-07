"""Profile every stage of the deterministic local preview pipeline."""

from __future__ import annotations

import argparse
import os
import statistics
import time
from pathlib import Path

import cv2
import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap, QTransform
from PySide6.QtWidgets import QApplication, QLabel

from app.compositing import AlphaFaceCompositor
from app.pipeline.types import FrameFormat, TrackingStatus, VideoFrame
from app.reference import PoseSpaceReferenceSelector, ReferenceLibraryStore
from app.rendering import GeometricFaceRenderer
from app.stabilization import TemporalStabilizer
from app.tracking import MediaPipeFaceTracker


def summary(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    return statistics.fmean(values), float(np.percentile(values, 95))


def print_stage(name: str, values: list[float]) -> None:
    mean, p95 = summary(values)
    print(f"{name:<22} mean {mean:7.2f} ms; p95 {p95:7.2f} ms")


def frame_image(frame: VideoFrame) -> QImage:
    height, width = frame.rgb.shape[:2]
    return QImage(
        frame.rgb.data,
        width,
        height,
        int(frame.rgb.strides[0]),
        QImage.Format.Format_RGB888,
    ).copy()


def main() -> int:
    repository = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--clip", type=Path, default=repository / "tests" / "assets" / "controlled_motion.mp4"
    )
    parser.add_argument("--references", type=Path, default=repository / "assets" / "pose_guides")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--max-frames", type=int, default=90)
    parser.add_argument("--warmup-frames", type=int, default=10)
    parser.add_argument("--preview-width", type=int, default=960)
    parser.add_argument("--preview-height", type=int, default=540)
    parser.add_argument("--no-ui", action="store_true")
    args = parser.parse_args()
    if (
        args.width <= 0
        or args.height <= 0
        or args.preview_width <= 0
        or args.preview_height <= 0
        or args.max_frames <= args.warmup_frames
        or args.warmup_frames < 0
    ):
        parser.error("dimensions must be positive and max frames must exceed warmup frames")

    capture = cv2.VideoCapture(str(args.clip))
    if not capture.isOpened():
        print(f"Could not open clip: {args.clip}")
        return 2
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    if source_fps <= 0:
        source_fps = 30.0

    app = QApplication.instance() or QApplication(["facelive-preview-benchmark"])
    label = QLabel()
    label.resize(args.preview_width, args.preview_height)

    library = ReferenceLibraryStore()
    tracker = MediaPipeFaceTracker()
    renderer = GeometricFaceRenderer()
    compositor = AlphaFaceCompositor()
    selector = PoseSpaceReferenceSelector()
    stabilizer = TemporalStabilizer()
    stage: dict[str, list[float]] = {
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
    measured = 0
    processed = 0
    no_face = 0
    started = 0.0
    frame_period_ns = round(1_000_000_000 / source_fps)

    try:
        for slot in library.slots():
            library.add_reference(slot.slot_id, args.references / f"{slot.slot_id}.png")
        references = library.references()
        tracker.open()
        renderer.open(FrameFormat(args.width, args.height, source_fps))

        for offset in range(args.max_frames):
            full_started = time.perf_counter_ns()
            capture_started = full_started
            ok, bgr = capture.read()
            if not ok:
                break
            if bgr.shape[1] != args.width or bgr.shape[0] != args.height:
                bgr = cv2.resize(bgr, (args.width, args.height), interpolation=cv2.INTER_LINEAR)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
            rgb.setflags(write=False)
            capture_finished = time.perf_counter_ns()
            timestamp_ns = 1_000_000_000 + offset * frame_period_ns
            frame = VideoFrame(offset, timestamp_ns, rgb)

            tracking_started = time.perf_counter_ns()
            if not tracker.submit(frame):
                raise RuntimeError("Tracker rejected a sequential benchmark frame")
            result = None
            deadline = time.monotonic() + 2.0
            while result is None and time.monotonic() < deadline:
                result = tracker.poll_latest()
                if result is None:
                    time.sleep(0.0005)
            if result is None:
                raise RuntimeError(f"Tracker timed out on frame {offset}")
            tracking_finished = time.perf_counter_ns()

            smoothing_started = time.perf_counter_ns()
            if result.status is TrackingStatus.TRACKED:
                face = stabilizer.update(result.face)
            elif result.status is TrackingStatus.NO_FACE:
                face = stabilizer.coast(result.frame_id, result.timestamp_ns)
            else:
                face = None
            smoothing_finished = time.perf_counter_ns()

            selection_started = time.perf_counter_ns()
            weights = selector.select(face, references) if face is not None else ()
            if face is not None:
                weights = stabilizer.smooth_weights(weights, face.timestamp_ns)
            selection_finished = time.perf_counter_ns()

            render_started = time.perf_counter_ns()
            rendered = renderer.render(face, references, weights) if face is not None else None
            render_finished = time.perf_counter_ns()

            composite_started = time.perf_counter_ns()
            output = compositor.composite(frame, rendered) if rendered is not None else frame
            composite_finished = time.perf_counter_ns()

            ui_copy_started = time.perf_counter_ns()
            image = frame_image(output)
            ui_copy_finished = time.perf_counter_ns()
            present_started = time.perf_counter_ns()
            if not args.no_ui:
                display_image = image.transformed(
                    QTransform().scale(-1.0, 1.0), Qt.TransformationMode.FastTransformation
                )
                pixmap = QPixmap.fromImage(display_image).scaled(
                    label.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
                label.setPixmap(pixmap)
                app.processEvents()
            present_finished = time.perf_counter_ns()

            if offset == args.warmup_frames:
                started = time.perf_counter()
            if offset < args.warmup_frames:
                continue
            measured += 1
            processed += int(face is not None)
            no_face += int(face is None)
            spans = {
                "capture_convert": (capture_started, capture_finished),
                "tracking": (tracking_started, tracking_finished),
                "stabilization": (smoothing_started, smoothing_finished),
                "reference_selection": (selection_started, selection_finished),
                "rendering": (render_started, render_finished),
                "compositing": (composite_started, composite_finished),
                "ui_copy": (ui_copy_started, ui_copy_finished),
                "ui_presentation": (present_started, present_finished),
                "full_frame": (full_started, present_finished),
            }
            for name, (span_started, span_finished) in spans.items():
                stage[name].append((span_finished - span_started) / 1_000_000)

        elapsed = time.perf_counter() - started if measured else 0.0
        print(f"Input: {args.clip}")
        print(f"Format: {args.width}x{args.height}; measured frames: {measured}")
        print(f"Processed: {processed}; no face: {no_face}")
        print(f"Throughput: {measured / elapsed:.2f} FPS" if elapsed > 0 else "Throughput: n/a")
        for name, values in stage.items():
            print_stage(name, values)
        print("Neural inference       n/a (geometric backend selected)")
        print("Queue policy           sequential benchmark; production stages are latest/one-frame")
        return 0
    finally:
        capture.release()
        renderer.close()
        tracker.close()
        library.close()
        label.clear()


if __name__ == "__main__":
    raise SystemExit(main())
