# FaceLive

Local Windows 11 x64 desktop application for a consenting user's prepared reference
appearance. The intended pipeline preserves the live body and surroundings while
replacing only the facial region.

**Current implementation: Milestones 2 through 9 — stabilized geometric replacement,
bounded real-time preview, and a versioned shared-memory frame transport.**
The app tracks one face asynchronously, displays a diagnostic mesh, pose, confidence,
expressions, and latency. Its reference dialog
enrolls the eight initial pose/expression images into a persistent local library. Continuous
pose-space weights drive a deterministic landmark warp and feathered facial composite.
Temporal smoothing stabilizes pose, landmarks, expressions, and reference weights; short
tracking gaps are held briefly. Original, diagnostic, and processed previews are available.
The geometric renderer remains the default. A tested optional neural-renderer adapter and
configuration boundary are available, but no large model, provider, or weights are bundled.
The profiled geometric preview sustains approximately 29 FPS at 1280×720 on the target
development machine using bounded latest-frame scheduling.
Completed processed frames can be published through a three-slot shared-memory ring to the
standalone test consumer. The future native virtual camera will use this documented ABI.
Virtual-camera output remains inactive.

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
- [Milestone 7 neural refinement evaluation and adapter proof of concept](docs/milestone_7.md)
- [Milestone 8 real-time profiling, optimization, and benchmark](docs/milestone_8.md)
- [Milestone 9 shared-memory transport and stress results](docs/milestone_9.md)
- [Native frame transport protocol v1](docs/frame_transport_protocol.md)

All processing is intended to stay local. Future functionality is for the user's
own face or explicitly consenting subjects. Identity verification, liveness/proctoring
bypass, audio modification, background replacement, and cloud inference are outside scope.
Continue only after a specific instruction for the next milestone.
