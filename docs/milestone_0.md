# Milestone 0 report

## Delivered

Bootstrapped an initially empty repository with a Python 3.12 environment, locked
dependencies, a PySide6 application shell, validated TOML configuration, bounded UTC
logging, all eight requested protocols, shared data records, tests, lint/format
configuration, and architecture/developer documentation. Initialized local Git
metadata; no commit or remote was created.

The idle shell opens no devices and publishes no frames. Configuration and logs work
independently of the launch directory. There is no face replacement, neural model,
native camera implementation, or installer in this milestone.

## Changed files (all new)

Repository and setup:

```text
.gitattributes
.gitignore
.python-version
pyproject.toml
uv.lock
README.md
config/default.toml
```

Application, protocols, and data contracts:

```text
app/__init__.py
app/__main__.py
app/config.py
app/main.py
app/camera/__init__.py
app/camera/protocol.py
app/compositing/__init__.py
app/compositing/protocol.py
app/diagnostics/__init__.py
app/diagnostics/logging_setup.py
app/pipeline/__init__.py
app/pipeline/protocol.py
app/pipeline/types.py
app/reference/__init__.py
app/reference/protocol.py
app/rendering/__init__.py
app/rendering/protocol.py
app/stabilization/__init__.py
app/stabilization/protocol.py
app/tracking/__init__.py
app/tracking/protocol.py
app/ui/__init__.py
app/ui/main_window.py
```

Tests, documentation, and reserved component directories:

```text
tests/unit/test_config.py
tests/unit/test_logging.py
tests/integration/test_startup.py
tests/assets/README.md
docs/architecture.md
docs/development.md
docs/reference_capture.md
docs/milestone_0.md
models/README.md
native/virtual_camera/README.md
tools/capture_references/README.md
tools/benchmarks/README.md
```

Generated local state (ignored, not project source): `.bootstrap/`, `.venv/`, Python
bytecode and pytest/Ruff caches. uv also provisions a user-local Python runtime and
dependency cache. Application smoke runs create the documented rotating log file.

## Validation

Validated on Windows 11 x64 with Python 3.12.15, PySide6 6.11.2, NumPy 2.5.3,
pytest 9.1.1, and Ruff 0.16.10 (exact dependencies in `uv.lock`).

| Command/check | Result |
| --- | --- |
| `.\.bootstrap\Scripts\uv.exe sync --locked` | Locked environment installs successfully |
| `.\.bootstrap\Scripts\uv.exe pip check` | All installed packages compatible |
| `.\.venv\Scripts\python.exe -m pytest` | 25 passed |
| `.\.venv\Scripts\python.exe -m ruff check .` | Passed |
| `.\.venv\Scripts\python.exe -m ruff format --check .` | Passed |
| `.\.venv\Scripts\python.exe -m compileall -q app` | Passed |
| `.\.venv\Scripts\facelive.exe --config config/default.toml --smoke-test` | Native Windows Qt shell started and closed, exit 0 |
| Native widget screenshot inspection | Idle shell/controls/metrics legible, menu contrast corrected |

The integration tests also run `python -m app` and `python -m app.main` from a temporary
directory with Qt's offscreen backend, confirm logging and clean event-loop shutdown,
verify inactive controls, and check that invalid configuration fails before UI startup.
Logs were written successfully to the configured local directory. No hardware or models
were opened. Visual inspection used an ignored screenshot in `.bootstrap/`.

## Stable decisions and assumptions

- Python 3.12, Windows 11 x64, PySide6 desktop UI; processing remains local.
- RGB uint8 full-frame arrays, pixel coordinates, explicit degree/sign conventions,
  frame ID plus monotonic capture timestamp propagated through the pipeline.
- Read-only published buffers, explicit ownership, bounded queues, newest complete
  frame preference, and exact tracking/frame pairing.
- Backend-independent protocols; asynchronous tracker submit/poll; Qt confined to UI.
- Cached reference preprocessing and continuous selection; deterministic rendering first.
- Placeholder output on missing/failed processing, with a visible local reason;
  no automatic source or identity switch.
- Native output is a later standalone FrameSink adapter with its own versioned ABI.

See [architecture.md](architecture.md) for full method contracts and decisions that
must be preserved or explicitly revised with evidence. Exact landmark topology,
reference manifest, freshness thresholds, native ABI, and model versions remain open.

## Limitations and next step

Only the shell runs. Protocols specify behavior but have no camera/tracker/render/sink
implementations or runtime conformance checks. Camera negotiation, processing quality,
720p/30 FPS, latency, dropped-frame performance, and meeting-app compatibility cannot
be evaluated yet; their metrics are not applicable at this milestone. No face/video
fixtures are included because nothing processes video yet.

Recommended next milestone: physical camera enumeration/capture, bounded preview,
disconnect handling, and capture diagnostics. Stop here until explicitly instructed.
