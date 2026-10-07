# Milestone 4 — Pose-Space Reference Selection

## Delivered

- `PoseSpaceReferenceSelector`, a concrete implementation of the existing
  `ReferenceSelector` protocol.
- Continuous piecewise-linear yaw interpolation across available zero-pitch anchors.
- Continuous pitch influence toward the available up or down axis references.
- Expression interpolation at an exact pose, including front-neutral/front-smile using
  live mouth-smile blendshapes.
- Finite, nonnegative, normalized `ReferenceWeight` outputs in stable library order.
- Graceful selection from incomplete libraries by using the remaining pose anchors.
- A live two-column bar visualization for every configured reference slot, including
  current yaw, pitch, and smile diagnostics.

Selection runs after an asynchronous tracking result is matched to its exact camera frame.
It reads cached `PreparedReference` metadata and does not process reference pixels again.
No rendered face or composite image is produced.

## Interpolation rules

The initial reference set forms a cross in pose space:

```text
                         up (0, -15)
                              |
right-40 — right-20 — front (0, 0) — left-20 — left-40
                              |
                        down (0, +15)
```

Yaw uses the two adjacent anchors surrounding the tracked yaw. For example, yaw `-10°`
produces equal front and right-20 weights; yaw `-30°` produces equal right-20 and right-40
weights. Values outside the captured range clamp to the outer reference.

Pitch transfers a continuous fraction of the current yaw weights to the corresponding
pitch axis. At yaw `-10°`, pitch `+7.5°`, the neutral pose weights are 25% front, 25%
right-20, and 50% down. This represents the sparse references honestly until combined
yaw-and-pitch captures are introduced.

Expression is evaluated within each selected pose. At the front pose, a 75% smile signal
produces 25% front-neutral and 75% front-smile. At poses without a smile variant, their
pose weight remains on the available neutral reference. Missing blendshape values are not
treated as measured activity.

Roll and apparent face scale remain untouched in `FaceState`. Later rendering geometry can
rotate and resize the selected face without requiring roll or distance reference slots.

## Validation

Unit tests cover exact anchors, outer boundaries, all yaw midpoints, positive and negative
pitch, mixed yaw/pitch, smile interpolation, roll/scale independence, incomplete libraries,
empty libraries, duplicate IDs, nonnegative weights, and normalization. The application
integration test verifies the visualization is initialized and remains explicit when no
references are enrolled.

```powershell
.\.bootstrap\Scripts\uv.exe run --locked pytest
.\.bootstrap\Scripts\uv.exe run --locked ruff check .
.\.bootstrap\Scripts\uv.exe run --locked ruff format --check .
$env:QT_QPA_PLATFORM = 'offscreen'
.\.venv\Scripts\python.exe -m app --smoke-test
Remove-Item Env:QT_QPA_PLATFORM
```

## Change inventory

```text
README.md
app/main.py
app/pipeline/types.py
app/reference/__init__.py
app/reference/protocol.py
app/reference/selector.py
app/ui/main_window.py
app/ui/reference_weights.py
docs/architecture.md
docs/development.md
docs/milestone_4.md
tests/integration/test_startup.py
tests/unit/test_reference_selector.py
```

## Decisions to preserve

- Reference pose signs match `FaceState`: positive yaw points toward image right/the
  subject's left, and positive pitch points down.
- Selection operates on cached metadata and emits weights only. It does not own image
  rendering, geometric transformation, or compositing.
- Adjacent pose transitions remain continuous and normalized; do not replace them with
  hard nearest-reference switching.
- Expression variants divide a pose's existing weight rather than changing pose geometry.
- Roll and scale remain geometric parameters outside pose-space selection.
- Diagnostic percentages are rounded for display; pipeline weights retain floating-point
  precision and are normalized independently of the visualization.
