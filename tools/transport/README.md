# Frame transport tools

Run the standalone consumer while FaceLive is producing processed frames:

```powershell
.\.venv\Scripts\python.exe tools\transport\consume_frames.py `
  --name facelive_frames_v1 --seconds 10
```

The consumer attaches independently of the PySide6 application, always copies the newest
complete slot, verifies CRC-32 when present, reports skipped transport sequences, and
reconnects after producer shutdown or restart. Validation errors produce a nonzero exit
code. It does not register a virtual camera.

Use `tools/benchmarks/frame_transport.py` for a self-contained producer/consumer stress run.
