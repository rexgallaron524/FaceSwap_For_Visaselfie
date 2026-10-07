# Milestone 5 — Geometric Face Rendering and Compositing

## Delivered

- `GeometricFaceRenderer`, implementing the existing `FaceRenderer` protocol.
- Schema-checked Delaunay facial topology with piecewise affine landmark warping.
- Continuous pixel interpolation from normalized `ReferenceWeight` values.
- Translation, apparent scale, yaw/pitch deformation, and roll alignment through the live
  landmark mesh.
- A face-oval convex-hull mask with scale-relative inward feathering.
- `AlphaFaceCompositor`, with local RGB mean/contrast correction and immutable output.
- Explicit **Original camera**, **Diagnostic tracking**, and **Processed output** preview
  modes.
- Live timing rows for tracking, rendering, compositing, and complete matched-frame work.

The prototype replaces only the facial region. Body, hair outside the face oval, clothing,
and background come from the matching original camera frame. It does not use a rectangular
overlay, neural portrait model, or virtual-camera output.

## Processing flow

1. The asynchronous tracker returns a `FaceState` for its exact retained `VideoFrame`.
2. Pose/expression selection produces normalized weights for enrolled references.
3. The renderer verifies that live and reference meshes use
   `mediapipe-face-landmarker-478-v1`.
4. A cached facial triangle topology maps each selected reference onto the live landmarks.
5. Warped triangle pixels are combined using the continuous reference weights.
6. The live face oval creates a feathered full-frame alpha mask.
7. The compositor matches local color and brightness, then blends into a new RGB frame.

Processed mode shows a clear unavailable message when no reference or valid composite
exists. It does not silently substitute the original camera image.

## Validation and observed performance

Deterministic tests verify topology generation, continuous two-reference interpolation,
feathered nonrectangular alpha, immutable outputs, exact preservation outside alpha, frame
identity matching, and rejection of incompatible inputs. Existing camera, tracking,
reference, selection, persistence, responsive-layout, and startup tests remain active.

A synthetic CPU benchmark used a 1280×720 frame and an approximately 330×390 face region:

| Active references | Median rendering | Median compositing | Median total |
| ---: | ---: | ---: | ---: |
| 1 | 51.7 ms | 23.3 ms | 76.2 ms |
| 2 | 59.3 ms | 20.6 ms | 79.7 ms |
| 4 | 77.8 ms | 20.4 ms | 99.4 ms |

These are development-machine synthetic measurements, not a guaranteed camera rate. Live
results depend on face size, the number of nonzero reference weights, CPU, and camera load.

## Decisions to preserve

- Rendering and compositing preserve `frame_id` and `timestamp_ns`; mismatches fail rather
  than combining geometry with an unrelated camera frame.
- Canonical frames remain unmirrored. Mirroring is still a local display transform after
  processing.
- Renderer output was initially full-frame RGB plus float32 alpha. Milestone 8 extends the
  contract with an explicit `active_region`, allowing unambiguous tightly cropped arrays
  while the compositor still owns placement.
- Landmark schema compatibility is explicit. Do not infer topology from landmark count.
- Selection owns pose/expression weights; rendering consumes them and does not implement a
  second selection policy.
- Roll and scale are geometric effects derived from live landmarks, not reference slots.
- Pixels with zero alpha remain identical to the original frame.
- This deterministic renderer is a replaceable protocol implementation and must not become
  coupled to future neural or virtual-camera backends.

## Commands

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m ruff format --check .
$env:QT_QPA_PLATFORM = 'offscreen'
.\.venv\Scripts\python.exe -m app --smoke-test
Remove-Item Env:QT_QPA_PLATFORM
```
