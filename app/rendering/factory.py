"""Composition helpers for selecting a FaceRenderer backend."""

from __future__ import annotations

from collections.abc import Callable

from app.config import RendererConfig
from app.rendering.geometric import GeometricFaceRenderer
from app.rendering.neural import NeuralFaceRenderer, PythonModulePortraitRuntime
from app.rendering.protocol import FaceRenderer


def create_renderer_factory(config: RendererConfig) -> Callable[[], FaceRenderer]:
    """Return a renderer factory without loading an optional model runtime."""
    if config.backend == "geometric":
        return GeometricFaceRenderer

    def create_neural_renderer() -> FaceRenderer:
        runtime = PythonModulePortraitRuntime(
            config.runtime_module, config.model_directory, config.device
        )
        return NeuralFaceRenderer(runtime)

    return create_neural_renderer
