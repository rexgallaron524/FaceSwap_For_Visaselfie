# Milestone 2 — Live Face Tracking and Diagnostics

## Delivered

- `MediaPipeFaceTracker`, a `FaceTracker` implementation using Face Landmarker in
  `LIVE_STREAM` mode with one primary face.
- A backend-independent `FaceState` with frame identity, capture timestamp, pixel bounds,
  pixel and normalized 3D landmarks, center, apparent scale, yaw/pitch/roll, derived
  tracking confidence, and normalized blendshape values.
- A bounded asynchronous handoff: only one frame may be in flight. Busy submissions are
  dropped and counted, so camera input cannot create a growing tracking queue.
- Exact result matching by frame ID and timestamp. The UI retains the accepted frame and
  displays a result only with that matching image.
- Optional local diagnostics showing the face region, sampled landmarks, center, pose,
  confidence, smile, left/right eye closure, mouth opening, tracking latency, and drops.
- Clean no-face and tracking-error states. Camera preview remains usable if the tracking
  model cannot start.
- Strictly increasing camera timestamps even on Windows clocks that return the same tick
  for adjacent captured frames.

The overlay is drawn on a UI-owned image copy after the mirror-preview transform. Tracking
always receives the original unmirrored RGB `VideoFrame`, and no face pixels are modified.

## Scheduling and measurements

MediaPipe calls the adapter back from its live-stream worker. The adapter converts the
result immediately to immutable internal records and publishes one latest result under a
lock. Tracking latency measures submission to callback completion with the monotonic clock.

On the development Integrated Webcam, a six-second 1280×720 run negotiated 30 FPS and
measured 30.0 capture FPS. It completed 142 tracking requests and dropped 37 submissions
while the tracker was busy. Mean tracking latency was 13.9 ms and p95 was 16.0 ms. Face
availability during that run depended on the subject being in view.

A repeatable static-image exercise at 512×512 completed 60 tracked frames at 78.8 FPS,
with 11.2 ms mean latency and 16.0 ms p95 latency. These figures describe this development
machine and are not device guarantees.

MediaPipe does not expose a per-result face-presence score. The displayed confidence is a
documented geometry-quality signal: the square root of the in-frame landmark ratio times a
face-size quality term that reaches one at 18% of the frame. The UI labels it **derived** so
it is not presented as a probability.

## Validation

Run from the repository root after the locked environment is synchronized:

```powershell
.\.bootstrap\Scripts\uv.exe run --locked pytest
.\.bootstrap\Scripts\uv.exe run --locked ruff check .
.\.bootstrap\Scripts\uv.exe run --locked ruff format --check .
$env:QT_QPA_PLATFORM = 'offscreen'
.\.venv\Scripts\python.exe -m app --smoke-test
Remove-Item Env:QT_QPA_PLATFORM
```

Unit coverage includes normalized-to-pixel conversion, landmark bounds, known rotation
recovery, confidence behavior, blendshape name normalization, asynchronous backpressure,
tracked/no-face conversion, timestamps, and camera abstraction behavior. The integration
startup test injects deterministic camera and tracker implementations and verifies rendered
diagnostics without changing source pixels.

## Change inventory

```text
README.md
app/camera/opencv_source.py
app/main.py
app/pipeline/types.py
app/reference/detector.py
app/tracking/__init__.py
app/tracking/geometry.py
app/tracking/mediapipe_tracker.py
app/ui/main_window.py
docs/architecture.md
docs/development.md
docs/milestone_2.md
docs/milestone_3.md
models/README.md
tests/integration/test_startup.py
tests/unit/test_camera_source.py
tests/unit/test_face_tracker.py
tests/unit/test_tracking_geometry.py
```

## Decisions to preserve

- Camera and pipeline frames remain unmirrored RGB; mirroring and diagnostics are local
  preview transforms.
- `FaceState` is the only tracking data consumed outside adapters. MediaPipe objects must
  not cross that boundary.
- Landmark schema names and pose signs are compatibility contracts. Change either only
  with an explicit migration of consumers and cached reference data.
- Asynchronous results must match both frame ID and timestamp. Keep retention bounded and
  drop work under pressure instead of accumulating latency.
- Missing blendshapes remain absent rather than being reported as measured zero.
- Tracking confidence remains explicitly identified by its derivation unless a later
  backend provides a documented confidence measurement.

Face rendering, blending, stabilization, and virtual-camera output are outside this
milestone.
