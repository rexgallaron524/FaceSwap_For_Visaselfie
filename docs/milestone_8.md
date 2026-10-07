# Milestone 8 — real-time performance optimization

## Acceptance threshold and machine

For this milestone, “approximately 30 FPS” means at least 27 processed presentations per
second, or 90% of the requested 30 FPS, over 150 measured frames after a 30-frame warmup.
The benchmark uses the production tracker, selector, stabilizer, renderer, compositor, and
Qt image presentation path. It expands the repository's deterministic 640×360 controlled
clip to 1280×720 before processing.

Measurements below were taken locally on:

- Windows 11 Pro build 26200;
- Intel Core i5-1135G7, 4 cores / 8 logical processors;
- Python 3.12.15;
- OpenCV 5.0.0;
- geometric renderer, with no neural provider or inference stage.

## Reproduce

From the repository root:

```powershell
.\.venv\Scripts\python.exe tools\benchmarks\preview_pipeline.py `
  --max-frames 75 --warmup-frames 15

.\.venv\Scripts\python.exe tools\benchmarks\realtime_preview.py `
  --frames 180 --warmup-frames 30
```

The first command is a sequential stage profiler. The second exercises the bounded,
overlapped production schedule at 1280×720 and 30 FPS. Use `--save-frame output.png` for a
visual output sample. The real-time command defaults to the measured production settings:
10 tracking updates per second, four OpenCV threads, and a 192-pixel internal face working
resolution.

## Baseline and result

The initial sequential profile processed 60 measured frames at **4.32 FPS**. Rendering and
compositing dominated at 134.81 ms and 46.39 ms mean respectively; full-frame work averaged
235.06 ms. This measurement established the optimization targets before changes.

Three final validation runs produced **28.59 FPS**, **28.98 FPS**, and **29.38 FPS**. The
table records the last run, where full-frame latency includes capture conversion.

| Stage | Mean | p95 |
| --- | ---: | ---: |
| Capture and RGB conversion | 4.70 ms | 7.56 ms |
| Tracking | 27.16 ms | 32.00 ms |
| Stabilization | 3.14 ms | 5.79 ms |
| Reference selection | 0.18 ms | 0.30 ms |
| Face rendering | 11.61 ms | 20.02 ms |
| Compositing | 4.49 ms | 6.93 ms |
| UI image copy | 1.58 ms | 2.37 ms |
| UI presentation | 7.25 ms | 9.73 ms |
| Capture to presentation | 40.34 ms | 65.51 ms |

The run presented 148 of 150 measured input frames. Two stale pending processing requests
were replaced, 100 input frames were intentionally skipped by the 10 FPS tracking cadence,
and the tracker reported no busy drops. Queue bounds were one tracker frame, one active plus
one replaceable pending processing request, and one UI presentation. Resource shutdown took
1.91 seconds in the recorded run; earlier final runs took 1.85–3.33 seconds. All
released the capture, executors, renderer caches, and MediaPipe instances without a live
process or retained camera handle.

A separate five-second production `CameraSource` check opened the Integrated Webcam at
1280×720/30, observed 30.29 capture FPS, delivered 30.40 polling FPS, and skipped no frames.

## Changes justified by profiling

- Reference appearances are aligned once and cached at a bounded working resolution.
- The renderer blends cached appearances first, then performs one smooth sparse-landmark
  remap instead of hundreds of triangle warps.
- Rendering and mask generation operate at a 192-pixel internal face resolution and scale
  only the facial crop to its final size. The camera frame and final composite remain
  1280×720.
- `RenderedFace.active_region` permits tightly cropped RGB and alpha arrays. The compositor
  validates and processes only this region, avoiding full-frame zero buffers and scans.
- OpenCV color statistics and blending replace large temporary NumPy expressions.
- Rendering and compositing run in one worker outside the Qt thread. One newest pending
  request replaces older pending work; queues cannot grow.
- Tracking runs at 10 FPS while the latest stabilized state drives current camera frames at
  up to 30 FPS. This preserves current body/background pixels while avoiding CPU contention
  between MediaPipe, OpenCV, and Qt.
- Qt scales before mirroring and uses its fast preview transform. This changes local display
  scaling only; canonical output pixels are unchanged.
- Four OpenCV threads performed better than one, two, or the library default on this CPU.

An attempted external resize before MediaPipe tracking increased latency and reduced output
throughput, so it was removed. ONNX Runtime, DirectML, CUDA, and TensorRT were not introduced:
the selected renderer has no neural inference stage, and profiling identified deterministic
warping, compositing, scheduling, and UI scaling as the bottlenecks.

## Remaining limits

The benchmark uses a deterministic prerecorded clip and synthetic consent-oriented guide
assets, so camera-driver cost and live lighting are not represented. Tracking refreshes at
10 FPS; very fast head motion may reveal short pose holds between updates. p95 full-frame
latency remains about two 30 FPS frame periods when a newest pending request is retained.
Performance settings are configurable because machines with fewer cores or a neural
provider require a new profile rather than copying these numbers.

This milestone meets the local processed-preview threshold. It does not add a neural model
or virtual-camera output.
