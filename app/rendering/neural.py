"""Replaceable neural-renderer adapter and optional local runtime loader."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from math import isfinite
from pathlib import Path
from time import perf_counter_ns
from typing import Protocol

import numpy as np

from app.pipeline.types import (
    FaceState,
    FrameFormat,
    PreparedReference,
    ReferenceWeight,
    RenderedFace,
    StageError,
)

_WEIGHT_TOLERANCE = 1e-4


@dataclass(frozen=True, slots=True)
class WeightedAppearance:
    """An opaque, precomputed reference appearance and its current selection weight."""

    reference_id: str
    appearance: object
    weight: float


@dataclass(frozen=True, slots=True)
class NeuralRendererMetrics:
    """Most recent adapter timings plus cumulative cache activity."""

    preparation_ms: float = 0.0
    inference_ms: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0


class PortraitAnimationRuntime(Protocol):
    """Model-specific runtime hidden behind the FaceRenderer adapter."""

    def open(self, output_format: FrameFormat) -> None: ...

    def prepare_reference(self, reference: PreparedReference) -> object:
        """Compute reusable appearance state for one enrolled reference."""
        ...

    def render(self, face: FaceState, appearances: tuple[WeightedAppearance, ...]) -> RenderedFace:
        """Infer a full-frame RGB face layer and facial alpha mask."""
        ...

    def release_reference(self, appearance: object) -> None: ...

    def close(self) -> None: ...


class PythonModulePortraitRuntime:
    """Load an optional local model provider without importing it in the UI layer.

    The configured module must expose ``create_runtime(model_directory=..., device=...)``.
    The returned object implements :class:`PortraitAnimationRuntime`. This small provider
    boundary allows LivePortrait to run in a separately maintained package or IPC client.
    """

    def __init__(self, module_name: str, model_directory: Path, device: str) -> None:
        self._module_name = module_name
        self._model_directory = model_directory
        self._device = device
        self._runtime: PortraitAnimationRuntime | None = None

    def open(self, output_format: FrameFormat) -> None:
        if self._runtime is not None:
            self.close()
        try:
            module = import_module(self._module_name)
        except (ImportError, ModuleNotFoundError) as exc:
            raise StageError(
                f"Neural runtime module '{self._module_name}' is not installed. "
                "Use renderer.backend = 'geometric' or install the configured local provider."
            ) from exc
        factory = getattr(module, "create_runtime", None)
        if not callable(factory):
            raise StageError(
                f"Neural runtime module '{self._module_name}' has no callable create_runtime"
            )
        try:
            runtime = factory(model_directory=self._model_directory, device=self._device)
            missing = [
                name
                for name in ("open", "prepare_reference", "render", "release_reference", "close")
                if not callable(getattr(runtime, name, None))
            ]
            if missing:
                raise StageError(
                    "Neural runtime is missing required operation(s): " + ", ".join(missing)
                )
            runtime.open(output_format)
        except StageError:
            raise
        except Exception as exc:
            raise StageError(f"Could not initialize neural runtime: {exc}") from exc
        self._runtime = runtime

    def _require_runtime(self) -> PortraitAnimationRuntime:
        if self._runtime is None:
            raise StageError("Neural runtime is not open")
        return self._runtime

    def prepare_reference(self, reference: PreparedReference) -> object:
        try:
            return self._require_runtime().prepare_reference(reference)
        except StageError:
            raise
        except Exception as exc:
            raise StageError(
                f"Neural reference preparation failed for {reference.reference_id}: {exc}"
            ) from exc

    def render(self, face: FaceState, appearances: tuple[WeightedAppearance, ...]) -> RenderedFace:
        try:
            return self._require_runtime().render(face, appearances)
        except StageError:
            raise
        except Exception as exc:
            raise StageError(f"Neural inference failed: {exc}") from exc

    def release_reference(self, appearance: object) -> None:
        runtime = self._runtime
        if runtime is None:
            return
        try:
            runtime.release_reference(appearance)
        except Exception:
            # Cleanup is best effort and must not mask a camera/library transition.
            pass

    def close(self) -> None:
        runtime, self._runtime = self._runtime, None
        if runtime is None:
            return
        try:
            runtime.close()
        except Exception:
            # FaceRenderer.close() is intentionally idempotent and non-failing.
            pass


@dataclass(slots=True)
class _CachedAppearance:
    signature: tuple[object, ...]
    value: object


class NeuralFaceRenderer:
    """Adapt a portrait-animation runtime to the stable FaceRenderer protocol.

    Every enrolled reference is preprocessed on first use and cached until that reference
    object is replaced or the renderer closes. The runtime remains responsible for model
    semantics and placement, while this adapter owns validation, cache lifetime, and timing.
    """

    def __init__(self, runtime: PortraitAnimationRuntime) -> None:
        self._runtime = runtime
        self._format: FrameFormat | None = None
        self._cache: dict[str, _CachedAppearance] = {}
        self._cache_hits = 0
        self._cache_misses = 0
        self._metrics = NeuralRendererMetrics()

    @property
    def metrics(self) -> NeuralRendererMetrics:
        return self._metrics

    @property
    def prepared_reference_count(self) -> int:
        return len(self._cache)

    def open(self, output_format: FrameFormat) -> None:
        self.close()
        self._runtime.open(output_format)
        self._format = output_format
        self._cache_hits = 0
        self._cache_misses = 0
        self._metrics = NeuralRendererMetrics()

    def _release(self, entry: _CachedAppearance) -> None:
        self._runtime.release_reference(entry.value)

    @staticmethod
    def _reference_signature(reference: PreparedReference) -> tuple[object, ...]:
        # ReferenceLibraryStore returns a fresh PreparedReference wrapper on each snapshot,
        # while the expensive image and landmark objects remain stable until enrollment
        # changes. Include semantic fields so future slot edits invalidate the cache too.
        return (
            id(reference.rgb),
            id(reference.landmarks),
            reference.rgb.shape,
            reference.yaw,
            reference.pitch,
            reference.expression,
            reference.landmark_schema,
        )

    def _synchronize_cache(self, references: tuple[PreparedReference, ...]) -> float:
        started = perf_counter_ns()
        active_ids = {reference.reference_id for reference in references}
        for reference_id in tuple(self._cache):
            if reference_id not in active_ids:
                self._release(self._cache.pop(reference_id))

        for reference in references:
            cached = self._cache.get(reference.reference_id)
            signature = self._reference_signature(reference)
            unchanged = cached is not None and cached.signature == signature
            if unchanged:
                self._cache_hits += 1
                continue
            if cached is not None:
                self._release(cached)
            value = self._runtime.prepare_reference(reference)
            self._cache[reference.reference_id] = _CachedAppearance(signature, value)
            self._cache_misses += 1
        return (perf_counter_ns() - started) / 1_000_000

    def render(
        self,
        face: FaceState,
        references: tuple[PreparedReference, ...],
        weights: tuple[ReferenceWeight, ...],
    ) -> RenderedFace:
        output_format = self._format
        if output_format is None:
            raise StageError("Neural face renderer is not open")
        if not references or not weights:
            raise StageError("Neural rendering requires references and normalized weights")
        by_id = {reference.reference_id: reference for reference in references}
        if len(by_id) != len(references):
            raise StageError("Reference IDs must be unique during neural rendering")
        total = sum(weight.weight for weight in weights)
        if not isfinite(total) or abs(total - 1.0) > _WEIGHT_TOLERANCE:
            raise StageError("Neural reference weights must be normalized")
        for weight in weights:
            if weight.reference_id not in by_id:
                raise StageError(f"Selected reference is unavailable: {weight.reference_id}")

        preparation_ms = self._synchronize_cache(references)
        selected = tuple(
            WeightedAppearance(
                weight.reference_id,
                self._cache[weight.reference_id].value,
                weight.weight,
            )
            for weight in weights
            if weight.weight > 0.0
        )
        if not selected:
            raise StageError("Neural rendering requires a positive reference weight")

        started = perf_counter_ns()
        rendered = self._runtime.render(face, selected)
        inference_ms = (perf_counter_ns() - started) / 1_000_000
        rendered = self._validated_output(rendered, face, output_format)
        self._metrics = NeuralRendererMetrics(
            preparation_ms, inference_ms, self._cache_hits, self._cache_misses
        )
        return rendered

    @staticmethod
    def _validated_output(
        rendered: RenderedFace, face: FaceState, output_format: FrameFormat
    ) -> RenderedFace:
        if not isinstance(rendered, RenderedFace):
            raise StageError("Neural runtime returned an invalid result type")
        if (rendered.frame_id, rendered.timestamp_ns) != (face.frame_id, face.timestamp_ns):
            raise StageError("Neural runtime returned stale frame identity")
        expected_rgb = (output_format.height, output_format.width, 3)
        expected_alpha = (output_format.height, output_format.width)
        if rendered.rgb.shape != expected_rgb or rendered.rgb.dtype != np.uint8:
            raise StageError("Neural runtime returned an invalid RGB frame")
        if rendered.alpha.shape != expected_alpha or rendered.alpha.dtype != np.float32:
            raise StageError("Neural runtime returned an invalid alpha mask")
        if not np.isfinite(rendered.alpha).all() or np.any(
            (rendered.alpha < 0.0) | (rendered.alpha > 1.0)
        ):
            raise StageError("Neural runtime alpha must contain finite values in [0, 1]")

        rgb = np.array(rendered.rgb, dtype=np.uint8, order="C", copy=True)
        alpha = np.array(rendered.alpha, dtype=np.float32, order="C", copy=True)
        rgb.setflags(write=False)
        alpha.setflags(write=False)
        return RenderedFace(face.frame_id, face.timestamp_ns, rgb, alpha)

    def close(self) -> None:
        for entry in self._cache.values():
            self._release(entry)
        self._cache.clear()
        self._runtime.close()
        self._format = None
