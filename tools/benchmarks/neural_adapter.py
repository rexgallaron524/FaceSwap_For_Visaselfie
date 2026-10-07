"""Measure NeuralFaceRenderer adapter/cache overhead without installing a neural model."""

from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import replace

import numpy as np

from app.pipeline.types import (
    FaceState,
    FrameFormat,
    HeadPose,
    NormalizedLandmark,
    Point2D,
    PreparedReference,
    Rect,
    ReferenceWeight,
    RenderedFace,
)
from app.rendering import NeuralFaceRenderer


class NullPortraitRuntime:
    """Contract-only runtime; intentionally performs no model inference."""

    def __init__(self) -> None:
        self.output_format: FrameFormat | None = None
        self.rgb: np.ndarray | None = None
        self.alpha: np.ndarray | None = None

    def open(self, output_format: FrameFormat) -> None:
        self.output_format = output_format
        self.rgb = np.zeros((output_format.height, output_format.width, 3), dtype=np.uint8)
        self.alpha = np.zeros((output_format.height, output_format.width), dtype=np.float32)

    def prepare_reference(self, reference: PreparedReference) -> object:
        # Keep a tiny stand-in for a model appearance tensor.
        return (reference.reference_id, float(reference.rgb.mean()))

    def render(self, face: FaceState, appearances) -> RenderedFace:
        assert self.rgb is not None and self.alpha is not None
        return RenderedFace(face.frame_id, face.timestamp_ns, self.rgb, self.alpha)

    def release_reference(self, appearance: object) -> None:
        pass

    def close(self) -> None:
        self.output_format = None
        self.rgb = None
        self.alpha = None


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((len(ordered) - 1) * fraction))
    return ordered[index]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=200)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()
    if args.frames <= 0 or args.width <= 0 or args.height <= 0:
        parser.error("frames, width, and height must be positive")

    runtime = NullPortraitRuntime()
    renderer = NeuralFaceRenderer(runtime)
    renderer.open(FrameFormat(args.width, args.height, 30))
    point = Point2D(args.width / 2, args.height / 2)
    normalized = NormalizedLandmark(0.5, 0.5, 0.0)
    references = tuple(
        PreparedReference(
            f"reference-{index}",
            0.0,
            0.0,
            "neutral",
            np.full((256, 256, 3), index * 24, dtype=np.uint8),
            (Point2D(128.0, 128.0),),
            "benchmark-v1",
        )
        for index in range(8)
    )
    weights = tuple(
        ReferenceWeight(item.reference_id, 1.0 / len(references)) for item in references
    )

    durations: list[float] = []
    inference_durations: list[float] = []
    cold_total_ms = 0.0
    cold_prepare_ms = 0.0
    try:
        for frame_id in range(args.frames + 1):
            tracked = FaceState(
                frame_id,
                1_000_000_000 + frame_id * 33_333_333,
                Rect(args.width * 0.35, args.height * 0.2, args.width * 0.3, args.height * 0.55),
                point,
                0.3,
                HeadPose(0.0, 0.0, 0.0),
                (point,),
                (normalized,),
                "benchmark-v1",
                1.0,
            )
            snapshot = tuple(replace(reference) for reference in references)
            started = time.perf_counter_ns()
            renderer.render(tracked, snapshot, weights)
            duration_ms = (time.perf_counter_ns() - started) / 1_000_000
            if frame_id == 0:
                cold_total_ms = duration_ms
                cold_prepare_ms = renderer.metrics.preparation_ms
            else:
                durations.append(duration_ms)
                inference_durations.append(renderer.metrics.inference_ms)
    finally:
        renderer.close()

    print("Neural adapter contract benchmark (model inference excluded)")
    print(f"Format: {args.width}x{args.height}; references: {len(references)}")
    print(f"Cold call: {cold_total_ms:.3f} ms; cache preparation: {cold_prepare_ms:.3f} ms")
    print(
        f"Warm adapter total: mean {statistics.fmean(durations):.3f} ms; "
        f"p95 {percentile(durations, 0.95):.3f} ms"
    )
    print(
        f"Null runtime call: mean {statistics.fmean(inference_durations):.3f} ms; "
        f"p95 {percentile(inference_durations, 0.95):.3f} ms"
    )
    print(f"Cached references: {len(references)}; warm cache misses: 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
