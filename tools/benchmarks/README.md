# Pipeline benchmarks

Add reproducible benchmarks with the processing pipeline. Report hardware, resolution,
duration, warmup, processed/output FPS, mean and tail latency, and frame drops per stage.
Milestone 1 includes `camera_capture.py` for the physical-camera stage:

```powershell
.\.venv\Scripts\python.exe tools\benchmarks\camera_capture.py --seconds 8
```

The command opens the selected physical camera through the production `CameraSource`
adapter and reports negotiated format, observed capture FPS, polling FPS, and skipped
frames. Close meeting/camera apps first so they do not hold the device.

Milestone 6 adds a full deterministic pipeline benchmark using the controlled motion clip
and the synthetic reference guides:

```powershell
.\.venv\Scripts\python.exe tools\benchmarks\temporal_pipeline.py
.\.venv\Scripts\python.exe tools\benchmarks\temporal_pipeline.py --disable-smoothing
```

Both commands exclude reference enrollment and the first 15 warmup frames. They report
pipeline/output FPS, tracking latency, processing latency, total latency, no-face frames,
and frames recovered by the bounded tracking hold.

Use `--start-frame 210 --max-frames 30 --warmup-frames 5` to isolate the controlled dropout
segment. `--max-frames 90 --warmup-frames 10` gives a shorter translation/pose comparison.

Milestone 7 adds a contract-only benchmark for the optional neural adapter. It validates
reference caching and measures full-frame adapter overhead without installing or pretending
to benchmark a neural model:

```powershell
.\.venv\Scripts\python.exe tools\benchmarks\neural_adapter.py
```

Actual model inference is reported by `NeuralFaceRenderer.metrics.inference_ms` when a local
provider is installed.

Milestone 8 adds a sequential profiler and a bounded real-time preview benchmark:

```powershell
.\.venv\Scripts\python.exe tools\benchmarks\preview_pipeline.py `
  --max-frames 75 --warmup-frames 15

.\.venv\Scripts\python.exe tools\benchmarks\realtime_preview.py `
  --frames 180 --warmup-frames 30
```

`preview_pipeline.py` identifies expensive stages without scheduling overlap.
`realtime_preview.py` uses the production one-frame tracker, one active plus one newest
pending processing request, and offscreen Qt presentation. It reports mean and p95 capture,
tracking, stabilization, selection, rendering, compositing, UI, and full-frame timings plus
all drop counts and resource-shutdown time. Use `--save-frame output.png` to inspect one
composite. `--tracking-fps`, `--opencv-threads`, and `--working-resolution` support explicit
comparison runs; their defaults match the profiled application configuration.

Recorded Milestone 8 hardware, commands, rejected experiments, and results are in
[`docs/milestone_8.md`](../../docs/milestone_8.md).
