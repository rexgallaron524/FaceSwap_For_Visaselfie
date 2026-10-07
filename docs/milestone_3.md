# Milestone 3 — reference library and enrollment

## Delivered

- Eight required, data-driven slots: front neutral, front smile, left 20°, left 40°,
  right 20°, right 40°, up, and down.
- A detector protocol and lazy MediaPipe Face Landmarker still-image adapter.
- Enrollment validation for readable dimensions, exactly one face, complete finite
  landmarks, face size/coverage, exposure, sharpness, and reliable eye alignment.
- One-time conversion to immutable 512×512 RGB, 478 aligned landmarks, 52 blendshape
  values, source digest, quality measurements, and normalized face bounds.
- A versioned JSON manifest with deterministic serialization and neighboring PNG cache,
  including compatibility checks, path containment, and SHA-256 asset verification.
- An extensible slot model that can add required or optional poses and expressions without
  changing the `ReferenceLibrary` protocol or prepared-reference pipeline record.
- A PySide6 enrollment dialog with per-slot instructions, image selection/removal,
  normalized thumbnails, validity status, completion count, and save/load actions.
- Automatic saving after each accepted edit, startup restoration of the last active
  library, a per-user default location, **Save as…**, and visible storage status.
- Capture guidance, model provenance, architecture notes, and locked dependencies.

Face replacement, live face tracking, reference selection, compositing, and virtual-camera
output are not enabled by this milestone.

## Verification

The deterministic suite uses generated images and a fake detector. It covers preprocessing
success and rejection paths, immutable normalized output, canonical metadata serialization,
save/load and lookup order, asset corruption, duplicate/unknown slots, optional slot
extension, application startup, and the eight-card enrollment dialog.

```text
python -m pytest -q
52 passed

python -m ruff check .
All checks passed!
```

The production MediaPipe adapter was also exercised with a known face image on the
development machine:

```text
first preprocessing pass (model initialization included): 692.8 ms
second preprocessing pass:                                 22.8 ms
output: 512×512 RGB, 478 landmarks, 52 blendshapes
```

A physical camera smoke check negotiated 1280×720 at 30 FPS. Its empty-scene frame was
correctly rejected because no face was detected. Enrollment work is triggered only when a
user chooses an image and is not part of the live frame loop.

## Changed files

```text
.gitignore
README.md
app/config.py
app/reference/__init__.py
app/reference/detector.py
app/reference/library.py
app/reference/model.py
app/reference/preprocessor.py
app/reference/session.py
app/ui/main_window.py
app/ui/reference_dialog.py
docs/architecture.md
docs/development.md
docs/milestone_3.md
docs/reference_capture.md
models/README.md
models/face_landmarker.task
models/LICENSE.face-landmarker.txt
pyproject.toml
tests/integration/test_startup.py
tests/unit/test_reference_library.py
tests/unit/test_reference_preprocessor.py
tests/unit/test_reference_session.py
uv.lock
```

## Stable decisions for later milestones

- Canonical frame/reference pixels remain unmirrored, contiguous RGB `uint8`. Mirroring is
  only a local preview transform.
- Slot IDs are persistent keys. Positive yaw means the subject turns left in an unmirrored
  image; positive pitch looks down.
- Manifest schema/version, normalization size, and landmark schema are compatibility
  boundaries. Change them only with an explicit migration or cache rebuild.
- Original source images are not copied. The manifest and its asset directory form one
  library and must move together.
- Expensive face detection and alignment happen during enrollment. Future per-frame code
  consumes cached `PreparedReference` records through the existing library interface.
- MediaPipe remains behind `ReferenceFaceDetector`; its objects and normalized coordinates
  do not leak into pipeline records.
- A failed library load clears existing references so stale identity data cannot continue.
- Automated validation establishes technical image usability. Pose accuracy and subject
  consistency remain enrollment responsibilities until an explicit pose/identity validator
  is introduced.
