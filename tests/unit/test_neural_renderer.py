from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from app.config import RendererConfig
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
    StageError,
)
from app.rendering import (
    GeometricFaceRenderer,
    NeuralFaceRenderer,
    PythonModulePortraitRuntime,
    create_renderer_factory,
)


def face(frame_id: int = 7) -> FaceState:
    return FaceState(
        frame_id,
        2_000_000_000 + frame_id,
        Rect(2.0, 1.0, 4.0, 4.0),
        Point2D(4.0, 3.0),
        0.5,
        HeadPose(5.0, -2.0, 1.0),
        (Point2D(4.0, 3.0),),
        (NormalizedLandmark(0.5, 0.5, 0.0),),
        "test-v1",
        0.9,
        {"jaw_open": 0.2},
    )


def reference(reference_id: str, value: int) -> PreparedReference:
    rgb = np.full((4, 4, 3), value, dtype=np.uint8)
    rgb.setflags(write=False)
    return PreparedReference(
        reference_id, 0.0, 0.0, "neutral", rgb, (Point2D(2.0, 2.0),), "test-v1"
    )


class FakeRuntime:
    def __init__(self, *, stale: bool = False) -> None:
        self.output_format: FrameFormat | None = None
        self.prepared: list[str] = []
        self.released: list[object] = []
        self.selections: list[tuple[tuple[str, float], ...]] = []
        self.stale = stale

    def open(self, output_format: FrameFormat) -> None:
        self.output_format = output_format

    def prepare_reference(self, prepared_reference: PreparedReference) -> object:
        self.prepared.append(prepared_reference.reference_id)
        return (prepared_reference.reference_id, id(prepared_reference))

    def render(self, tracked_face, appearances):
        assert self.output_format is not None
        self.selections.append(tuple((item.reference_id, item.weight) for item in appearances))
        rgb = np.zeros((self.output_format.height, self.output_format.width, 3), dtype=np.uint8)
        alpha = np.zeros((self.output_format.height, self.output_format.width), dtype=np.float32)
        output_id = tracked_face.frame_id - 1 if self.stale else tracked_face.frame_id
        return RenderedFace(output_id, tracked_face.timestamp_ns, rgb, alpha)

    def release_reference(self, appearance: object) -> None:
        self.released.append(appearance)

    def close(self) -> None:
        self.output_format = None


def test_neural_adapter_caches_all_reference_appearances_and_preserves_weights():
    runtime = FakeRuntime()
    renderer = NeuralFaceRenderer(runtime)
    renderer.open(FrameFormat(8, 6, 30))
    references = (reference("front", 20), reference("right", 80))
    weights = (ReferenceWeight("front", 0.75), ReferenceWeight("right", 0.25))

    first = renderer.render(face(), references, weights)
    # The library publishes fresh PreparedReference wrappers while retaining the same
    # prepared image/landmark objects. This must remain a cache hit.
    refreshed_snapshot = tuple(replace(item) for item in references)
    second = renderer.render(face(8), refreshed_snapshot, weights)

    assert first.rgb.shape == (6, 8, 3)
    assert second.alpha.shape == (6, 8)
    assert not first.rgb.flags.writeable
    assert not first.alpha.flags.writeable
    assert runtime.prepared == ["front", "right"]
    assert runtime.selections == [
        (("front", 0.75), ("right", 0.25)),
        (("front", 0.75), ("right", 0.25)),
    ]
    assert renderer.prepared_reference_count == 2
    assert renderer.metrics.cache_misses == 2
    assert renderer.metrics.cache_hits == 2
    assert renderer.metrics.preparation_ms >= 0.0
    assert renderer.metrics.inference_ms >= 0.0

    changed = (references[0], reference("right", 120))
    renderer.render(face(9), changed, weights)
    assert runtime.prepared == ["front", "right", "right"]
    assert len(runtime.released) == 1

    renderer.close()
    assert renderer.prepared_reference_count == 0
    assert len(runtime.released) == 3


def test_neural_adapter_rejects_stale_runtime_output():
    renderer = NeuralFaceRenderer(FakeRuntime(stale=True))
    renderer.open(FrameFormat(8, 6, 30))
    with pytest.raises(StageError, match="stale frame identity"):
        renderer.render(face(), (reference("front", 20),), (ReferenceWeight("front", 1.0),))


def test_optional_runtime_reports_actionable_missing_module_error(tmp_path):
    runtime = PythonModulePortraitRuntime(
        "facelive_missing_test_runtime", tmp_path / "models", "cpu"
    )
    with pytest.raises(StageError, match="is not installed"):
        runtime.open(FrameFormat(8, 6, 30))


def test_renderer_factory_selects_configured_backend(tmp_path):
    geometric = create_renderer_factory(RendererConfig())()
    neural = create_renderer_factory(
        RendererConfig(backend="liveportrait", model_directory=tmp_path)
    )()

    assert isinstance(geometric, GeometricFaceRenderer)
    assert isinstance(neural, NeuralFaceRenderer)


def test_neural_adapter_requires_normalized_available_weights():
    renderer = NeuralFaceRenderer(FakeRuntime())
    renderer.open(FrameFormat(8, 6, 30))
    tracked = face()
    enrolled = (reference("front", 20),)

    with pytest.raises(StageError, match="normalized"):
        renderer.render(tracked, enrolled, (ReferenceWeight("front", 0.5),))
    with pytest.raises(StageError, match="unavailable"):
        renderer.render(
            tracked,
            enrolled,
            (ReferenceWeight("missing", 1.0),),
        )


def test_neural_adapter_validates_alpha_values():
    class InvalidAlphaRuntime(FakeRuntime):
        def render(self, tracked_face, appearances):
            result = super().render(tracked_face, appearances)
            alpha = np.full(result.alpha.shape, 1.5, dtype=np.float32)
            return replace(result, alpha=alpha)

    renderer = NeuralFaceRenderer(InvalidAlphaRuntime())
    renderer.open(FrameFormat(8, 6, 30))
    with pytest.raises(StageError, match="finite values"):
        renderer.render(face(), (reference("front", 20),), (ReferenceWeight("front", 1.0),))
