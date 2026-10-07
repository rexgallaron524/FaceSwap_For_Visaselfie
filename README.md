# FaceLive

Local Windows 11 x64 desktop application for a consenting user's prepared reference
appearance. The intended pipeline preserves the live body and surroundings while
replacing only the facial region.

**Current milestone: 3 — reference library and enrollment.** The app captures and previews
physical-camera frames, and its reference-library dialog enrolls the eight initial pose and
expression images. Enrollment validates one usable face, aligns it to a normalized 512×512
RGB image, records reusable landmarks and metadata, and saves a versioned library locally.
Face replacement and virtual-camera output remain inactive.

## Quick start (PowerShell)

With `uv` installed:

```powershell
uv sync --locked
uv run --locked facelive
```

Python 3.12 is selected by `.python-version`; uv can provision it automatically.
See [development setup](docs/development.md) for setup with the existing Python launcher,
local bootstrap commands, and a pip alternative.

```powershell
uv run --locked pytest
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked facelive --smoke-test
```

## Documentation

- [Architecture and stable contracts](docs/architecture.md)
- [Environment, configuration, logging, and validation](docs/development.md)
- [Reference capture and enrollment guide](docs/reference_capture.md)
- [Milestone 0 change inventory and validation](docs/milestone_0.md)
- [Milestone 1 change inventory, tests, and observed performance](docs/milestone_1.md)
- [Milestone 3 change inventory, tests, and observed performance](docs/milestone_3.md)

All processing is intended to stay local. Future functionality is for the user's
own face or explicitly consenting subjects. Identity verification, liveness/proctoring
bypass, audio modification, background replacement, and cloud inference are outside scope.
Continue only after a specific instruction for the next milestone.
