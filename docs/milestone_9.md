# Milestone 9 — frame transport boundary

## Result

Milestone 9 implements a versioned, bounded shared-memory transport behind the existing
`FrameSink` protocol. The PySide6 application opens the producer with the camera pipeline
and publishes each completed processed RGB frame from the processing worker. With no
consumer, publication reports `False` and avoids copying pixels. The local preview continues
independently.

The consumer is a separate Python implementation of the same ABI. It always reads the
newest valid sequence and returns an immutable local pixel copy. Windows named mutexes guard
the header and each slot, preventing partial payload exposure without making the producer
wait for consumer work.

The complete native contract is [frame_transport_protocol.md](frame_transport_protocol.md).

## Configuration

The default configuration enables:

```toml
[transport]
enabled = true
name = "facelive_frames_v1"
slot_count = 3
capacity_width = 1280
capacity_height = 720
consumer_timeout_ms = 2000
checksum = true
```

Capacity fixes the allocation size while active dimensions may be smaller. Increasing
capacity or changing slot count requires every process holding the old mapping to stop.

## Standalone validation

Start FaceLive, choose **Processed output**, then run:

```powershell
.\.venv\Scripts\python.exe tools\transport\consume_frames.py `
  --name facelive_frames_v1 --seconds 10
```

The diagnostics panel shows whether the consumer is connected and the producer-side
transport time. The standalone consumer reports sessions, newest sequence, stale sequence
gaps, capture-to-consumer frame age, and validation errors.

Run the self-contained stress benchmark:

```powershell
.\.venv\Scripts\python.exe tools\benchmarks\frame_transport.py `
  --frames 600 --consumer-delay-ms 1
```

On the Milestone 8 development machine, the 1280×720 RGB24 run produced:

| Measurement | Result |
| --- | ---: |
| Producer attempts | 600 |
| Successful publications | 584 |
| Nonblocking drops | 16 |
| Accepted publication rate | 423.13 FPS |
| Publication mean / p95 | 2.20 / 3.31 ms |
| Consumer frames / stale sequences skipped | 225 / 359 |
| Consumer frame-age mean / p95 | 7.49 / 16.00 ms |
| Corrupt or partially written frames | 0 |
| Logical shared-memory size | 8,294,848 bytes |

The consumer delay intentionally makes it slower than the producer. Sequence gaps prove
that stale frames are skipped instead of accumulating. The benchmark transfers more than
1.6 GB of frame payload during the measured run.

## Validation

| Check | Result |
| --- | --- |
| `python -m ruff check app tests tools` | Passed |
| `python -m pytest -q` | 114 passed |
| `python -m app --smoke-test` | Started and shut down cleanly |
| Standalone consumer smoke run | 24 complete frames, 1.33 ms mean age, no read errors |

## Reliability coverage

Automated tests cover:

- exact 256-byte header and 64-byte slot layouts;
- no-consumer behavior;
- newest-only reads and immutable copied pixels;
- clean producer restart with a new session and sequence reset;
- rejection of a second live producer or consumer;
- replacement of a terminated producer while a consumer retains the mapping;
- cross-process contention over 1,000 patterned frames;
- monotonic sequences, CRC verification, drop accounting, and detection of mixed pixels.

No Media Foundation source, registration, installer, or native pixel conversion is included
in this milestone.

