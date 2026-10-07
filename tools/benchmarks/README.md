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
