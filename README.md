# FaceLive

Local Windows 11 x64 desktop application for a consenting user's prepared reference
appearance. The intended pipeline preserves the live body and surroundings while
replacing only the facial region.

**Current implementation: Milestones 2 through 6 — stabilized geometric replacement.**
The app tracks one face asynchronously, displays a diagnostic mesh, pose, confidence,
expressions, and latency. Its reference dialog
enrolls the eight initial pose/expression images into a persistent local library. Continuous
pose-space weights drive a deterministic landmark warp and feathered facial composite.
Temporal smoothing stabilizes pose, landmarks, expressions, and reference weights; short
tracking gaps are held briefly. Original, diagnostic, and processed previews are available.
Neural rendering and virtual-camera output remain inactive.

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
- [Milestone 2 change inventory, tests, and observed performance](docs/milestone_2.md)
- [Milestone 3 change inventory, tests, and observed performance](docs/milestone_3.md)
- [Milestone 4 change inventory and interpolation rules](docs/milestone_4.md)
- [Milestone 5 geometric rendering, compositing, and performance](docs/milestone_5.md)
- [Milestone 6 temporal stability, expressions, and evaluation](docs/milestone_6.md)

All processing is intended to stay local. Future functionality is for the user's
own face or explicitly consenting subjects. Identity verification, liveness/proctoring
bypass, audio modification, background replacement, and cloud inference are outside scope.
Continue only after a specific instruction for the next milestone.
