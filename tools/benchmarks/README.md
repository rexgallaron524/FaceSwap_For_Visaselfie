# Pipeline benchmarks — reserved

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
