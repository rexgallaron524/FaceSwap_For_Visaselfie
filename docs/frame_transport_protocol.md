# FaceLive frame transport protocol v1

This document defines the native ABI between the Python desktop producer and the future
C++ Media Foundation consumer. The structures are fixed little-endian byte layouts. C++
must use explicit-width integer types and packing assertions rather than compiler-default
structure padding.

## Scope

- Windows 11, one producer, and one consumer in the same interactive logon session.
- One named shared-memory mapping containing a header and a fixed three-slot ring by
  default.
- Packed, top-down RGB24 frames. Pixel conversion for Media Foundation belongs in the
  native consumer.
- Newest-frame delivery. The protocol does not promise that every sequence reaches the
  consumer.
- No network, file, socket, or encoded-image transport.

The default mapping name is `facelive_frames_v1`. Configuration may change the name,
capacity, or slot count, but both processes must use the same ABI version and mapping.

## Integer and alignment rules

- All multibyte fields are unsigned little-endian integers.
- `uint16`, `uint32`, and `uint64` mean exactly 2, 4, and 8 bytes.
- The main header is 256 bytes.
- Each slot header is 64 bytes.
- Each slot stride is `align64(64 + capacity_width * capacity_height * 3)`.
- Slot `i` begins at `256 + i * slot_stride`.
- The logical mapping size is `256 + slot_count * slot_stride`. Windows may report a
  page-rounded physical view size; consumers validate that the view is at least the logical
  size declared in the header.
- Reserved bytes must be written as zero and ignored when read.

## Main header: 256 bytes

| Offset | Size | Type | Field | Meaning |
| ---: | ---: | --- | --- | --- |
| 0 | 8 | bytes | magic | ASCII `FLVCAM01` |
| 8 | 2 | uint16 | version | `1` |
| 10 | 2 | uint16 | header_size | `256` |
| 12 | 4 | uint32 | endian_marker | `0x01020304` |
| 16 | 8 | uint64 | total_size | Logical mapping size |
| 24 | 8 | uint64 | session_high | High half of producer UUID |
| 32 | 8 | uint64 | session_low | Low half of producer UUID |
| 40 | 4 | uint32 | capacity_width | Maximum payload width |
| 44 | 4 | uint32 | capacity_height | Maximum payload height |
| 48 | 4 | uint32 | width | Active session width |
| 52 | 4 | uint32 | height | Active session height |
| 56 | 4 | uint32 | stride | Active bytes per row; v1 requires `width * 3` |
| 60 | 4 | uint32 | pixel_format | `1` = RGB24 |
| 64 | 4 | uint32 | fps_numerator | Active frame-rate numerator |
| 68 | 4 | uint32 | fps_denominator | Active frame-rate denominator |
| 72 | 4 | uint32 | slot_count | 2 through 8; default 3 |
| 76 | 4 | uint32 | slot_header_size | `64` |
| 80 | 8 | uint64 | slot_stride | Aligned bytes per slot |
| 88 | 4 | uint32 | producer_pid | Zero after clean producer stop |
| 92 | 4 | uint32 | consumer_pid | Zero when the consumer detaches cleanly |
| 96 | 4 | uint32 | state | Lifecycle value below |
| 100 | 4 | uint32 | error_code | Zero in v1 normal operation |
| 104 | 8 | uint64 | producer_heartbeat_ns | Producer monotonic time |
| 112 | 8 | uint64 | consumer_heartbeat_ns | Consumer monotonic time |
| 120 | 8 | uint64 | consumer_session_high | Session acknowledged by consumer |
| 128 | 8 | uint64 | consumer_session_low | Session acknowledged by consumer |
| 136 | 8 | uint64 | published_sequence | Newest complete sequence; zero means none |
| 144 | 4 | uint32 | published_slot | Slot containing `published_sequence` |
| 148 | 4 | uint32 | reserved | Zero |
| 152 | 8 | uint64 | published_count | Successful publications in this session |
| 160 | 8 | uint64 | dropped_count | Rejected/no-consumer publications |
| 168 | 8 | uint64 | restart_count | Reuses of this still-mapped allocation |
| 176 | 80 | bytes | reserved | Zero |

Lifecycle values:

| Value | State | Meaning |
| ---: | --- | --- |
| 1 | `INITIALIZING` | Producer owns the mapping but slots are not ready |
| 2 | `RUNNING` | Header and slots can be consumed |
| 3 | `STOPPED` | Clean producer shutdown; wait for restart |
| 4 | `ERROR` | Producer declared a transport failure |

## Slot header: 64 bytes

| Offset | Size | Type | Field | Meaning |
| ---: | ---: | --- | --- | --- |
| 0 | 8 | uint64 | sequence | Transport sequence, starting at 1 per session |
| 8 | 8 | uint64 | session_high | Must equal main header session |
| 16 | 8 | uint64 | session_low | Must equal main header session |
| 24 | 8 | uint64 | timestamp_ns | Source `VideoFrame` monotonic capture timestamp |
| 32 | 8 | uint64 | frame_id | Source capture-session frame identifier |
| 40 | 4 | uint32 | payload_bytes | V1 requires `height * stride` |
| 44 | 4 | uint32 | width | Must equal active width |
| 48 | 4 | uint32 | height | Must equal active height |
| 52 | 4 | uint32 | stride | Must equal `width * 3` |
| 56 | 4 | uint32 | pixel_format | `1` = RGB24 |
| 60 | 4 | uint32 | crc32 | IEEE CRC-32; zero means omitted |
| 64 | payload_bytes | bytes | pixels | Top-down rows, RGB byte order |

The producer's Python timestamp uses `time.monotonic_ns()`. A native Windows process can
measure the same clock domain with `QueryPerformanceCounter` and
`QueryPerformanceFrequency`, converting ticks to nanoseconds with overflow-safe arithmetic.

## Synchronization objects

The protocol uses Windows named mutexes so neither language depends on undocumented atomic
behavior in a foreign runtime. Compute:

```text
digest = first 16 lowercase hex characters of SHA-256(UTF-8(mapping_name))
header mutex = Local\FaceLive.Transport.{digest}.Header
slot mutex i = Local\FaceLive.Transport.{digest}.Slot.{i}
```

The `Local\` namespace intentionally scopes v1 to the current terminal session. The future
native project must use matching names and the same user/session security boundary.

Frame publication and polling use zero-timeout mutex acquisition. They never wait for a
consumer or add work to a queue. `WAIT_ABANDONED` grants ownership; the next owner validates
or overwrites the protected data. Mutex release/acquire provides the required memory
ordering, so a consumer cannot observe a partial payload.

## Producer algorithm

1. Create or open the named mapping and mutexes.
2. Reject a header owned by a live producer PID.
3. For a stopped or dead producer, write a new random session UUID, reset sequences and
   counters, clear slot headers, then change state from `INITIALIZING` to `RUNNING`.
4. Before copying a frame, briefly read the header mutex. A consumer is connected only when
   its PID is alive, its acknowledged session matches, and its heartbeat is within the
   configured timeout. With no consumer, record a drop and avoid the image copy.
5. Try the next ring slot, then the remaining slots, using zero-timeout slot mutexes. If all
   are busy, record a drop.
6. While holding one slot mutex, write the slot header and payload and calculate CRC-32.
7. Acquire the header mutex without waiting. Publish `published_slot` and
   `published_sequence` only after the slot is complete. If the header is busy, leave the
   slot unpublished and record a drop.
8. Release the slot. The next successful publication receives the next sequence.

`FrameSink.publish()` returns `True` only when the header points to the completed slot.
Rejected attempts never consume a sequence number.

## Consumer algorithm

1. Open the mapping and validate magic, version, logical size, dimensions, format, slot
   count, and strides before reading pixels.
2. Under the header mutex, write the consumer PID, heartbeat, and acknowledged session;
   copy the newest sequence and slot index.
3. If the sequence was already returned for this session, return no frame.
4. Try the published slot mutex without waiting. If busy, return no frame and retry on the
   next poll. Do not fall back to an older slot.
5. Verify the slot sequence/session and all payload bounds, copy the pixels while holding
   the slot mutex, and verify CRC-32 when nonzero.
6. Release the mutex and expose the immutable local copy.

Sequence gaps are expected and count stale frames discarded by newest-frame delivery.

## Shutdown, crash, and restart

- Clean producer close writes `STOPPED`, clears `producer_pid`, and closes its handles.
- Clean consumer close clears its PID, heartbeat, and acknowledged session.
- If a consumer keeps the mapping open, a new producer reuses it, increments
  `restart_count`, assigns a new UUID, and resets sequence to 1. The UUID prevents sequence
  reset from being mistaken for stale data.
- A replacement producer may take over `INITIALIZING` or `RUNNING` state only when the
  recorded producer PID is no longer alive.
- If every process closes the Windows mapping, the kernel destroys it. A consumer must
  retry the mapping name when the producer later recreates it.
- A second live producer or consumer is rejected.

## C++ implementation requirements

- Assert main-header size 256 and slot-header size 64 at compile time.
- Do not reinterpret untrusted bytes before checking magic/version/size.
- Use `OpenFileMappingW`, `MapViewOfFile`, `CreateMutexW`/`OpenMutexW`, and
  `WaitForSingleObject(..., 0)` with the names above.
- Copy the slot payload before releasing its mutex. Media Foundation must own or copy that
  local buffer before the next poll.
- Keep RGB-to-NV12/BGRA conversion outside the mutex.
- Enforce producer heartbeat freshness and generate the future placeholder when stale.
- Treat any unsupported version, state, dimensions, bounds, format, or CRC as invalid input;
  never display bytes from a failed validation.

