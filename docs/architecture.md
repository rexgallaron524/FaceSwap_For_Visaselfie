# Architecture

## Status and scope

Milestone 0 implements the desktop shell, configuration, logging, and Python
protocols/data contracts. Everything in the processing diagram below is planned,
except the UI shell. Protocol definitions do not execute processing. Target platform:
Windows 11 x64, initial input/output 1280×720 at 30 FPS. This is a target, not a
measured capability of this milestone.

Python 3.12 is the initial tested minor version, pinned until the imaging/tracking
dependency set is validated. PySide6 supplies the desktop UI. NumPy supplies common
image types. OpenCV, MediaPipe, neural libraries, and native camera dependencies are
deferred until their milestones. Runtime imports never open hardware.

## Intended pipeline

```mermaid
flowchart LR
    Camera[Physical CameraSource] --> Frame[Bounded frame retention]
    Frame --> Tracker[Async FaceTracker]
    Tracker --> Stabilizer
    Stabilizer --> Selector[ReferenceSelector]
    Library[Prepared ReferenceLibrary] --> Selector
    Library --> Renderer[FaceRenderer]
    Selector --> Renderer
    Stabilizer --> Renderer
    Renderer --> Compositor
    Frame --> Compositor
    Compositor --> Policy[Output and failure policy]
    Policy --> Preview[Qt preview]
    Policy --> Sink[FrameSink]
    Sink --> Ring[Future bounded shared-memory buffer]
    Ring --> Native[Future native Media Foundation source]
    Native --> Meeting[Camera consumers]
```

The preview and sink must receive the same completed output image after failure
policy is applied. Physical input and virtual output have distinct identities and
lifecycles. Do not feed the FaceLive output back into its own input.

## Component boundaries

| Protocol | File | Responsibility |
| --- | --- | --- |
| `CameraSource` | `app/camera/protocol.py` | Enumerate, explicitly open physical input, negotiate format, retrieve newest frame, close |
| `FaceTracker` | `app/tracking/protocol.py` | Submit frames asynchronously and retrieve newest result without leaking backend types |
| `ReferenceLibrary` | `app/reference/protocol.py` | Load/validate/align/cache references once, expose immutable snapshots, clear |
| `ReferenceSelector` | `app/reference/protocol.py` | Produce continuous normalized weights for pose/expression |
| `FaceRenderer` | `app/rendering/protocol.py` | Render a face and mask; deterministic implementation first, swappable backend later |
| `Compositor` | `app/compositing/protocol.py` | Blend into the matching original frame, preserving pixels outside the mask |
| `Stabilizer` | `app/stabilization/protocol.py` | Smooth geometry/expression, preserve identity/time, reset history |
| `FrameSink` | `app/pipeline/protocol.py` | Accept completed frames without blocking or accumulating latency |

These are structural `typing.Protocol` interfaces, not inheritance requirements or
runtime validators. Implementations and boundary validation will accompany each stage.
The data records are in `app/pipeline/types.py`. Qt, MediaPipe, OpenCV, and Windows
objects must remain inside adapters. Stage failures use `StageError`; programming
errors must be logged and handled at the orchestration boundary as processing failures.

## Contracts to preserve

### Images and ownership

- Live images are unmirrored, C-contiguous NumPy `uint8` arrays, shape `(height,
  width, 3)`, RGB channel order. Camera adapters convert BGR; output adapters convert
  RGB to native formats. `FrameFormat` describes this fixed internal format.
- Alpha masks are full-frame `float32` arrays `(height, width)`, finite in `[0, 1]`.
  The renderer outputs full-frame RGB plus alpha to avoid ambiguous crop transforms.
  Full-frame allocation cost is an intentional initial simplicity tradeoff; profile
  before introducing ROI buffers.
- Frozen dataclasses prevent field rebinding; they do **not** freeze NumPy arrays or
  mappings. Producers must publish read-only snapshots (or enforce exclusive ownership),
  keep their buffers alive while referenced, and never overwrite published storage.
  Consumers must not mutate inputs. A retaining adapter must keep ownership or copy.
  Compositing produces a new frame and preserves original pixels wherever alpha is zero.
- Reference arrays and landmarks refer to aligned reference-image coordinates and
  remain cached until the library is cleared. Expensive preprocessing never runs per frame.

### Time, geometry, and expressions

- `frame_id` and `timestamp_ns` are strictly increasing within a capture session.
  Timestamp is capture time from the monotonic clock (`time.monotonic_ns`), not wall
  time. All downstream records preserve them. Restarting capture resets all histories
  and retained frames; IDs are not globally persistent identities.
- Async tracking results must match the retained original frame by **both** ID and
  timestamp. Never composite an old result onto the newest unrelated frame. Use bounded
  frame retention; discard results whose matching frames have already been dropped.
- `FaceState` uses full-frame pixel coordinates with origin at the upper left,
  x right, y down. Bounds are x/y/width/height. Scale is bounding width / image width.
  Coordinates can extend outside the frame; clipping belongs to render/composite adapters.
- Pose is in degrees: positive yaw turns toward image right (the subject's left in
  an unmirrored image); positive pitch looks down; positive roll is clockwise in the
  image. Roll and distance changes are geometric, not separate reference categories.
- Landmark ordering is described by an explicit versioned `landmark_schema`. Renderer
  and library must reject incompatible schemas instead of guessing by point count.
  No MediaPipe landmark ordering is hard-coded in core types.
- Blendshape coefficients are finite `[0, 1]`, with adapter-normalized canonical names
  such as `eye_blink_left`, `eye_blink_right`, `jaw_open`, `mouth_smile_left`, and
  `mouth_smile_right`; left/right identify the subject. Missing coefficients mean
  unavailable, not measured zero. Exact supported names and schema are finalized with
  the tracking milestone. Confidence `[0, 1]` must have a documented adapter derivation;
  do not imply a confidence score that the backend does not provide.
- Selection weights are finite, nonnegative, reference existing IDs, and sum to 1 for
  a nonempty result. Empty selection means no usable reference/unsupported pose.

### Scheduling and lifetime

- Qt widgets and the event loop stay on the GUI thread. Future capture and processing
  run outside it; signals deliver completed output to the UI. No processing workers
  are created in milestone 0.
- `CameraSource.read_latest`, `FaceTracker.submit/poll_latest`, and `FrameSink.publish`
  are nonblocking. Implementations use bounded storage and favor the newest complete
  frame. Resource setup/loading can block and belongs outside the GUI thread later.
- `submit(False)` explicitly reports a busy-frame drop. `poll_latest(None)` means no
  new result. `NO_FACE` is a completed tracking result, distinct from pending work.
  `TRACKED` carries a matching `FaceState`; `ERROR` carries an error and no face.
- Synchronous stage faults raise `StageError`; asynchronous faults return `ERROR`.
  Close/clear/reset operations are idempotent. Close stops worker callbacks before
  releasing buffers. Calls are serialized by the future pipeline controller; arbitrary
  concurrent public calls are not promised to be safe.
- Reset stabilization and drain pending work on source/library changes, tracking loss,
  and restart. Reconfiguration must quiesce callbacks before replacing references.
- Sink `False` means dropped/unconsumed, not a processing failure. Consumer disconnect
  does not stop preview. No unbounded queue, including the UI delivery queue, is allowed.

## Failure policy

Product decision: output a configured placeholder whenever a valid processed frame
cannot be produced while replacement is enabled. Never automatically fall back to
raw camera video or stale reference identity. The local UI shows the reason. A future
explicit user action to disable replacement may enable raw passthrough; enabling
passthrough and its visible state must be implemented together. Milestone 0 emits no video.

| Event | Intended behavior |
| --- | --- |
| Tracking lost / unsupported pose | Placeholder, show reason, reset smoothing |
| No library / library load fails | Placeholder; a failed load clears the previous subject |
| Physical camera disconnect | Placeholder; report input failure, no automatic source switch |
| Virtual consumer disconnect | Keep preview and processing; drop unconsumed frames |
| Renderer/compositor/model/GPU failure | Placeholder; report fault and release/restart affected stage explicitly |
| Malformed configuration / unavailable log directory | Fail startup with exit 2 and a diagnostic |
| Unexpected application failure | Log traceback, exit 1 |

Placeholder generation, recovery controller, freshness timeout, and IPC delivery are
future work. The native consumer must eventually enforce a producer heartbeat timeout
and emit its own placeholder when the producer dies; it cannot rely solely on Python.
Final timeout thresholds and reconnect UX need measurement and product review.

## Performance and future native boundary

Instrument capture, tracking, selection, rendering, compositing, and transfer separately.
Measure end-to-end age from capture timestamp, processed/output FPS, and dropped frames
by stage. The shell uses dashes rather than fabricated performance numbers.

After the preview pipeline is stable, implement a C++ Media Foundation custom Media
Source registered using `MFCreateVirtualCamera`, based on Microsoft's Windows Camera
sample. Keep all ML/image processing outside that layer. A versioned shared-memory
bounded ring will carry frame metadata and completed pixels. Sequence counters,
publication synchronization, heartbeat/fallback, access permissions, pixel conversion,
and ABI details require their own milestone. `FrameSink` is a Python abstraction,
**not** a shared-memory ABI. No native implementation exists yet.

## References

- [Qt for Python setup](https://doc.qt.io/qtforpython-6/gettingstarted.html)
- [MediaPipe Python setup](https://developers.google.com/edge/mediapipe/solutions/setup_python)
- [uv Python provisioning](https://docs.astral.sh/uv/guides/install-python/)

Neural refinement is deferred until deterministic geometry has been evaluated. The
next proposed milestone is physical camera input and bounded preview with capture
diagnostics; it requires an explicit user instruction.
