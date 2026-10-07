"""Validated local configuration; defaults do not depend on the working directory."""

from __future__ import annotations

import os
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
class AppConfig:
    video: VideoConfig = field(default_factory=VideoConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    debug: bool = False

    def __post_init__(self) -> None:
        if type(self.debug) is not bool:
            raise ConfigError("app.debug must be true or false")


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
    unknown = data.keys() - {"app", "video", "logging"}
    if unknown:
        raise ConfigError(f"Unknown configuration section(s): {', '.join(sorted(unknown))}")
    app = _table(data, "app", {"debug"})
    video = _table(data, "video", {"width", "height", "fps"})
    logging = _table(data, "logging", {"level", "directory", "max_bytes", "backup_count"})
    if "directory" in logging:
        raw = logging["directory"]
        if not isinstance(raw, str) or not raw.strip():
            raise ConfigError("logging.directory must be a nonempty path string")
        directory = Path(raw).expanduser()
        logging["directory"] = (
            directory if directory.is_absolute() else path.resolve().parent / directory
        )
    return AppConfig(video=VideoConfig(**video), logging=LoggingConfig(**logging), **app)
