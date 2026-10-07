# Milestone 1 report

## Delivered

- Named Windows webcam enumeration with stable per-device session IDs and explicit
  virtual-camera classification.
- `OpenCVCameraSource`, which negotiates the requested format, captures on a worker,
  timestamps successful frames, converts BGR to immutable contiguous RGB, and retains
  only the newest frame.
- Camera selection, refresh, asynchronous start, stop, aspect-preserving preview,
  negotiated-format display, measured capture FPS, and skipped-preview count.
- A **Mirror local preview** toggle, enabled by default, matching familiar video-call
  self-view behavior without changing capture or future outgoing-frame orientation.
- A modern high-contrast dark theme with distinct primary/secondary actions, readable
  disabled controls, clear focus/hover/pressed states, and color-coded session status.
- Clean failure UI for unavailable/open/disconnected cameras. Reconnect is explicit:
  reconnect the camera, choose Refresh, and start preview again. No raw source changes
  or automatic camera switching occur.
- Deterministic fake-camera tests and a physical-camera benchmark command. No personal
  image/video fixture was saved; the visual QA screenshot remains under ignored
  `.bootstrap/` and is not repository content.

No face tracking, face replacement, reference loading, virtual camera, or model was added.

## Changed files

```text
README.md
pyproject.toml
uv.lock
app/camera/__init__.py
app/camera/metrics.py
app/camera/opencv_source.py
app/main.py
app/pipeline/types.py
app/ui/main_window.py
docs/architecture.md
docs/development.md
docs/milestone_1.md
tests/integration/test_startup.py
tests/unit/test_camera_metrics.py
tests/unit/test_camera_source.py
tools/benchmarks/README.md
tools/benchmarks/camera_capture.py
```

## Validation and performance

Validated on Windows 11 x64 with Python 3.12.15, OpenCV 4.14.0, and an Integrated
Webcam. Exact package versions are locked in `uv.lock`.

| Check | Result |
| --- | --- |
| `python -m pytest` | 38 passed |
| `python -m ruff check .` | Passed |
| `python -m ruff format --check .` | Passed |
| `python -m compileall -q app tools/benchmarks/camera_capture.py` | Passed |
| `uv sync --locked` and `uv pip check` | Environment locked; packages compatible |
| 8-second standalone camera benchmark | 1280×720, 30.0 device-reported FPS, 27.55 observed FPS, 0 skipped |
| Automated native Qt preview | 1280×720, 25.7 observed FPS, visible frame, 0 skipped |
| High-contrast UI visual QA | Idle at 1120×740 and 900×700; live preview at 1120×740 |

The physical camera result is one run on available hardware, not a cross-device
guarantee. Exposure, lighting, USB bandwidth, drivers, and other camera consumers can
change FPS. The target is approximately 30 FPS where supported; this run met that intent.

## Stable decisions and limitations

- Core pipeline frames remain RGB; OpenCV BGR conversion stays inside its adapter.
- Core frames remain unmirrored. Mirroring is a local UI transform only, and the toggle
  never mutates or replaces the source frame used by future processing/output stages.
- Device setup and blocking frame reads stay off the Qt GUI thread.
- Camera buffering remains bounded to the newest complete frame. Frame ID gaps measure
  preview polling skips rather than allowing latency to accumulate.
- Capture timestamps use `monotonic_ns()` after successful camera reads. The source owns
  published immutable array storage until it replaces the latest frame; the UI copies it.
- The UI excludes recognized virtual cameras from physical input choices. Classification
  is name based because Windows/OpenCV do not expose a universal physical-device flag.
- Requested resolution/FPS are preferences; the actual frame size and reported FPS are
  shown after negotiation.
- Disconnect detection currently uses a two-second continuous read-failure threshold.
  Reconnection requires Refresh and Start preview; automatic reconnection is deferred.
- Device IDs are stable for one adapter session and derived from the enumerated device
  path. Persistent saved camera preference is not implemented.
- The supported minimum window size is 900×700 so camera controls, future controls, and
  session metrics remain legible without overlap.

Recommended next milestone: MediaPipe Face Landmarker integration producing the existing
backend-independent `FaceState`, with tracking overlays and stage timing. Stop here until
explicitly instructed.
