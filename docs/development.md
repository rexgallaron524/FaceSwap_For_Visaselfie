# Development

## Requirements

- Windows 11 x64; no GPU is needed. A webcam is needed only for live preview/benchmark.
- Python 3.12 x64. `.python-version` and `requires-python` intentionally select this
  minor version until later dependencies have been validated.
- Network access for initial dependency/runtime installation. The Face Landmarker model is
  versioned in `models/`, so enrollment and normal use stay local after checkout.

## Reproducible setup

Run from the repository root in PowerShell. If uv is already available:

```powershell
uv sync --locked
uv run --locked facelive
```

`uv.lock` pins runtime, development, and transitive dependencies. `uv sync` provisions
Python 3.12 if necessary and creates `.venv`. Check in lockfile changes with intentional
dependency updates; do not use upgrades as an implicit setup step.

The following installs uv in a separate local tooling environment when only the
Windows Python launcher is present. This is also how this workspace was bootstrapped:

```powershell
py -m venv .bootstrap
.\.bootstrap\Scripts\python.exe -m pip install uv==0.12.23
.\.bootstrap\Scripts\uv.exe sync --locked
.\.venv\Scripts\python.exe -m app
```

No activation or PowerShell execution-policy change is required. The local `.bootstrap`
and `.venv` directories are ignored. The managed Python runtime and download cache
may be stored in uv's user cache/data directories; existing Python installations are
not replaced. See [uv's runtime guide](https://docs.astral.sh/uv/guides/install-python/).

If Python 3.12 is already installed, a pip-only alternative is:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e . --group dev
.\.venv\Scripts\python.exe -m app
```

This pip alternative requires a pip version with dependency-group support and resolves
the ranges in `pyproject.toml`; it does not consume `uv.lock`. Use uv for the locked setup.

## Launch and configure

Equivalent launch entry points are `facelive`, `python -m app`, and `python -m app.main`
within the project environment. For example:

```powershell
.\.venv\Scripts\python.exe -m app --config config/default.toml
.\.venv\Scripts\python.exe -m app --smoke-test
```

`--smoke-test` shows the real shell, runs the event loop, then closes after 250 ms.
It does not open a camera. For headless environments:

```powershell
$env:QT_QPA_PLATFORM = 'offscreen'
.\.venv\Scripts\python.exe -m app --smoke-test
Remove-Item Env:QT_QPA_PLATFORM
```

With no `--config`, built-in defaults apply independently of the working directory.
`config/default.toml` is a documented example, not an automatically searched file.
Copy it to ignored `config/local.toml` for personal overrides. Unknown keys/sections,
invalid types, nonpositive dimensions/FPS/log limits, and unreadable files fail startup.
Only explicit configuration is loaded; no hidden environment-variable overrides exist
except standard `LOCALAPPDATA` for the log location and Qt's own environment settings.
Relative log paths resolve against the TOML file's parent. Video dimensions/FPS are
camera requests; the app shows the actual negotiated frame size and device-reported FPS.

Select **Manage references…** to open enrollment. Loading an image initializes MediaPipe
Face Landmarker lazily on the first request. Saving produces a manifest and asset directory
at the selected location; both are required when moving a library. Personal libraries and
source images belong in ignored `references/` or outside the repository. See the
[capture guide](reference_capture.md) for required poses and image conditions.

Logs go to `%LOCALAPPDATA%\FaceLive\logs\facelive.log` by default, with three backups
and a 2,000,000-byte rotation threshold. Without `LOCALAPPDATA`, the path is
`~/AppData/Local/FaceLive/logs`. Timestamps are UTC. Console output and the dedicated
`facelive` logger do not change root logging. Log events/diagnostics, never reference
images, frame pixels, or biometric coefficients. Exit codes: 0 normal, 2 configuration/
log setup/CLI failure, 1 unexpected app failure.

## Checks

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m ruff format --check .
.\.venv\Scripts\python.exe -m compileall -q app
```

Tests run the actual Qt process in offscreen mode with a temporary log directory.
They cover startup/clean shutdown, installed-package launch from another directory,
camera abstraction behavior, RGB conversion, timestamps, disconnect/reconnect UI state,
configuration validation, bounded logging, reference validation, deterministic manifest
serialization, checksum rejection, extensible slots, reference lookup, and the enrollment
dialog shell. Tests use generated NumPy frames and a fake detector, so no webcam, GPU, or
recorded face data is needed.

Run the optional physical-camera benchmark after closing other camera applications:

```powershell
.\.venv\Scripts\python.exe tools\benchmarks\camera_capture.py --seconds 8
```

Use `--device 1` for another enumerated physical camera. The tool reports negotiated
format, measured capture FPS, polling throughput, and skipped frames. Camera setup may
take a moment, but the app performs it away from the Qt GUI thread.

Use `python -m ruff format .` to format code. The lint rules cover errors, imports,
modern Python syntax, and common bug patterns. Test discovery is limited to `tests/`.

## Layout and workflow

`app/main.py` owns startup; `app/ui/` owns Qt; `app/camera/opencv_source.py` confines
OpenCV and Windows camera discovery; pipeline records and sink protocol live in
`app/pipeline/`. Each processing subsystem has a protocol module. `native/`,
fixture assets, and tooling directories contain scope notes until their milestones begin.
`app/reference/` owns enrollment models, the detector boundary, one-time preprocessing,
and persistence. `models/face_landmarker.task` is the pinned local enrollment model; its
provenance and checksum are in `models/README.md`. Read [architecture.md](architecture.md)
before changing contracts or manifest formats.

Inspect code before each milestone, state a plan, implement only that milestone,
add applicable tests, run checks, update docs, and stop for the next instruction.
The runtime must keep processing local. Do not add portrait-rendering or native-camera
packages until their dedicated milestones. No installer is produced in milestone 3.
