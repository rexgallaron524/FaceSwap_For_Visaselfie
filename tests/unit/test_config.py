from pathlib import Path

import pytest

from app.config import ConfigError, default_app_data_directory, load_config


def test_defaults_follow_local_app_data(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    config = load_config()
    assert default_app_data_directory() == tmp_path / "FaceLive"
    assert (config.video.width, config.video.height, config.video.fps) == (1280, 720, 30)
    assert config.logging.directory == tmp_path / "FaceLive" / "logs"
    assert config.renderer.backend == "geometric"
    assert config.renderer.model_directory == tmp_path / "FaceLive" / "models" / "liveportrait"
    assert config.debug is False


def test_partial_config_resolves_relative_paths(tmp_path):
    path = tmp_path / "settings.toml"
    path.write_text(
        '[app]\ndebug = true\n[logging]\ndirectory = "logs"\n'
        '[renderer]\nmodel_directory = "models/liveportrait"\n',
        encoding="utf-8",
    )
    config = load_config(path)
    assert config.debug is True
    assert config.video.fps == 30
    assert config.logging.directory == tmp_path / "logs"
    assert config.renderer.model_directory == tmp_path / "models" / "liveportrait"


@pytest.mark.parametrize(
    "content",
    [
        "[video]\nwidth = 0",
        "[video]\nheight = -1",
        "[video]\nfps = true",
        '[video]\nfps = "30"',
        "[video]\nfps = 29.97",
        "[video]\nwidht = 720",
        "[unknown]\nx = 1",
        "video = 123",
        '[app]\ndebug = "false"',
        '[logging]\nlevel = "TRACE"',
        "[logging]\nlevel = []",
        '[logging]\ndirectory = ""',
        "[logging]\ndirectory = false",
        "[logging]\nmax_bytes = 0",
        "[logging]\nbackup_count = -1",
        '[renderer]\nbackend = "unknown"',
        '[renderer]\nruntime_module = "bad-module"',
        '[renderer]\nmodel_directory = ""',
        "[renderer]\nmodel_directory = false",
        '[renderer]\ndevice = "directml"',
        "[renderer]\nunknown = true",
        "[invalid",
    ],
)
def test_invalid_configuration_is_rejected(tmp_path, content):
    path = tmp_path / "bad.toml"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(path)


def test_missing_explicit_config_is_an_error(tmp_path):
    with pytest.raises(ConfigError, match="Cannot read configuration"):
        load_config(tmp_path / "missing.toml")


def test_example_config_matches_defaults():
    example = Path(__file__).resolve().parents[2] / "config" / "default.toml"
    assert load_config(example) == load_config()
