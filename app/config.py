"""Validated local configuration; defaults do not depend on the working directory."""

from __future__ import annotations

import os
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    """The supplied configuration cannot be used."""


def default_app_data_directory() -> Path:
    base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
    return base / "FaceLive"


def default_log_directory() -> Path:
    return default_app_data_directory() / "logs"


def default_liveportrait_model_directory() -> Path:
    return default_app_data_directory() / "models" / "liveportrait"


def _positive_integer(name: str, value: object) -> None:
    if type(value) is not int or value <= 0:
        raise ConfigError(f"{name} must be a positive integer")


@dataclass(frozen=True, slots=True)
class VideoConfig:
    width: int = 1280
    height: int = 720
    fps: int = 30

    def __post_init__(self) -> None:
        for name in ("width", "height", "fps"):
            _positive_integer(f"video.{name}", getattr(self, name))


@dataclass(frozen=True, slots=True)
class LoggingConfig:
    level: str = "INFO"
    directory: Path = field(default_factory=default_log_directory)
    max_bytes: int = 2_000_000
    backup_count: int = 3

    def __post_init__(self) -> None:
        if self.level not in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
            raise ConfigError("logging.level must be DEBUG, INFO, WARNING, ERROR, or CRITICAL")
        if not isinstance(self.directory, Path):
            raise ConfigError("logging.directory must be a Path")
        _positive_integer("logging.max_bytes", self.max_bytes)
        _positive_integer("logging.backup_count", self.backup_count)


@dataclass(frozen=True, slots=True)
class RendererConfig:
    backend: str = "geometric"
    runtime_module: str = "facelive_liveportrait"
    model_directory: Path = field(default_factory=default_liveportrait_model_directory)
    device: str = "auto"

    def __post_init__(self) -> None:
        if self.backend not in ("geometric", "liveportrait"):
            raise ConfigError("renderer.backend must be geometric or liveportrait")
        if not isinstance(self.runtime_module, str) or not re.fullmatch(
            r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*", self.runtime_module
        ):
            raise ConfigError("renderer.runtime_module must be a dotted Python module name")
        if not isinstance(self.model_directory, Path):
            raise ConfigError("renderer.model_directory must be a Path")
        if self.device not in ("auto", "cpu", "cuda"):
            raise ConfigError("renderer.device must be auto, cpu, or cuda")


@dataclass(frozen=True, slots=True)
class PerformanceConfig:
    tracking_fps: int = 10
    opencv_threads: int = 4

    def __post_init__(self) -> None:
        _positive_integer("performance.tracking_fps", self.tracking_fps)
        _positive_integer("performance.opencv_threads", self.opencv_threads)


@dataclass(frozen=True, slots=True)
class TransportConfig:
    enabled: bool = True
    name: str = "facelive_frames_v1"
    slot_count: int = 3
    capacity_width: int = 1280
    capacity_height: int = 720
    consumer_timeout_ms: int = 2_000
    checksum: bool = True

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool or type(self.checksum) is not bool:
            raise ConfigError("transport.enabled and transport.checksum must be booleans")
        if not isinstance(self.name, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", self.name):
            raise ConfigError(
                "transport.name must use ASCII letters, digits, dot, dash, or underscore"
            )
        if type(self.slot_count) is not int or not 2 <= self.slot_count <= 8:
            raise ConfigError("transport.slot_count must be between 2 and 8")
        for name in ("capacity_width", "capacity_height", "consumer_timeout_ms"):
            _positive_integer(f"transport.{name}", getattr(self, name))


@dataclass(frozen=True, slots=True)
class AppConfig:
    video: VideoConfig = field(default_factory=VideoConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    renderer: RendererConfig = field(default_factory=RendererConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    transport: TransportConfig = field(default_factory=TransportConfig)
    debug: bool = False

    def __post_init__(self) -> None:
        if type(self.debug) is not bool:
            raise ConfigError("app.debug must be true or false")
        if self.performance.tracking_fps > self.video.fps:
            raise ConfigError("performance.tracking_fps cannot exceed video.fps")
        if self.transport.enabled and (
            self.video.width > self.transport.capacity_width
            or self.video.height > self.transport.capacity_height
        ):
            raise ConfigError("video dimensions exceed enabled transport capacity")


def _table(data: dict[str, Any], name: str, allowed: set[str]) -> dict[str, Any]:
    value = data.get(name, {})
    if not isinstance(value, dict):
        raise ConfigError(f"{name} must be a TOML table")
    unknown = value.keys() - allowed
    if unknown:
        raise ConfigError(f"Unknown {name} setting(s): {', '.join(sorted(unknown))}")
    return dict(value)


def load_config(path: Path | None = None) -> AppConfig:
    """Load an explicit TOML file; absent keys use defaults, unknown keys fail.

    Relative log directories resolve against the configuration file's directory.
    A missing explicitly requested file is an error, never a silent default.
    """
    if path is None:
        return AppConfig()
    try:
        with path.open("rb") as stream:
            data = tomllib.load(stream)
    except (OSError, ValueError) as exc:
        raise ConfigError(f"Cannot read configuration {path}: {exc}") from exc
    unknown = data.keys() - {
        "app",
        "video",
        "logging",
        "renderer",
        "performance",
        "transport",
    }
    if unknown:
        raise ConfigError(f"Unknown configuration section(s): {', '.join(sorted(unknown))}")
    app = _table(data, "app", {"debug"})
    video = _table(data, "video", {"width", "height", "fps"})
    logging = _table(data, "logging", {"level", "directory", "max_bytes", "backup_count"})
    renderer = _table(data, "renderer", {"backend", "runtime_module", "model_directory", "device"})
    performance = _table(data, "performance", {"tracking_fps", "opencv_threads"})
    transport = _table(
        data,
        "transport",
        {
            "enabled",
            "name",
            "slot_count",
            "capacity_width",
            "capacity_height",
            "consumer_timeout_ms",
            "checksum",
        },
    )
    if "directory" in logging:
        raw = logging["directory"]
        if not isinstance(raw, str) or not raw.strip():
            raise ConfigError("logging.directory must be a nonempty path string")
        directory = Path(raw).expanduser()
        logging["directory"] = (
            directory if directory.is_absolute() else path.resolve().parent / directory
        )
    if "model_directory" in renderer:
        raw = renderer["model_directory"]
        if not isinstance(raw, str) or not raw.strip():
            raise ConfigError("renderer.model_directory must be a nonempty path string")
        model_directory = Path(raw).expanduser()
        renderer["model_directory"] = (
            model_directory
            if model_directory.is_absolute()
            else path.resolve().parent / model_directory
        )
    return AppConfig(
        video=VideoConfig(**video),
        logging=LoggingConfig(**logging),
        renderer=RendererConfig(**renderer),
        performance=PerformanceConfig(**performance),
        transport=TransportConfig(**transport),
        **app,
    )
