"""Exercise the actual entry point and Qt event loop without camera hardware."""

import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("module", ["app", "app.main"])
def test_application_starts_and_exits_cleanly(tmp_path, module):
    # Run outside the repo to verify the installed package/defaults are CWD independent.
    result = subprocess.run(
        [sys.executable, "-m", module, "--smoke-test"],
        cwd=tmp_path,
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen", "LOCALAPPDATA": str(tmp_path)},
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    log = (tmp_path / "FaceLive" / "logs" / "facelive.log").read_text(encoding="utf-8")
    assert "Application shell ready; camera and output inactive" in log
    assert "Application stopped (exit 0)" in log


def test_invalid_config_fails_before_ui_startup(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "app", "--config", str(tmp_path / "missing.toml")],
        cwd=tmp_path,
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen", "LOCALAPPDATA": str(tmp_path)},
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 2
    assert "Cannot read configuration" in result.stderr
    assert not (tmp_path / "FaceLive" / "logs").exists()


def test_shell_controls_are_inactive(tmp_path):
    script = """
from app.config import AppConfig
from app.main import create_application
application, window = create_application(AppConfig())
window.show()
application.processEvents()
assert window.windowTitle() == 'FaceLive'
assert window.isVisible()
assert not window.camera_selector.isEnabled()
assert not window.load_references.isEnabled()
assert not window.replacement_toggle.isEnabled()
assert not window.replacement_toggle.isChecked()
assert not window.virtual_camera_button.isEnabled()
assert window.preview_title.text() == 'Preview is idle'
window.close()
application.processEvents()
assert not window.isVisible()
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(tmp_path),
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
