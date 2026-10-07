# Milestone 6 — Temporal Stability and Expression Handling

## Delivered

- `TemporalStabilizer`, implementing the existing stabilization boundary for live
  `FaceState` geometry and expressions.
- Time-based smoothing for translation, face bounds/scale, yaw, pitch, roll, pixel and
  normalized landmarks, reference weights, and blendshape values.
- Confidence-adaptive measurement strength and motion-adaptive response for large intended
  movements.
- Faster landmark and blendshape response around eyes and lips than the rest of the face.
- Asymmetric blink, mouth, and smile attack/release timing.
- A 150 ms bounded tracking hold that preserves current frame identity and decays dynamic
  mouth/eye expressions instead of extending stale activity.
- Blendshape-assisted eye closure, smile-corner motion, mouth opening, and a simple dark
  mouth cavity in the deterministic renderer.
- A **Temporal smoothing** comparison toggle plus smoothing state, latency, and correction
  diagnostics in the UI.
- A deterministic 240-frame controlled-motion MP4, its generator, and a full processing
  benchmark tool.

## Smoothing parameters

The filters use elapsed capture time rather than a fixed per-frame coefficient, so their
behavior remains comparable when camera or processed FPS changes.

| Signal | Time constant |
| --- | ---: |
| Translation | 65 ms |
| Scale/bounds | 90 ms |
| Yaw, pitch, roll | 80 ms |
| General landmarks | 55 ms |
| Eye and lip landmarks | 32 ms |
| Generic expressions | 75 ms |
| Smile attack / release | 90 / 140 ms |
| Blink attack / release | 18 / 45 ms |
| Mouth attack / release | 35 / 65 ms |
| Reference weights | 110 ms |
| Short tracking hold | 150 ms maximum |
| Long-gap reset | 750 ms |

Fast movement can raise the effective geometry coefficient by up to 45% of its remaining
range. Low confidence reduces measurement influence, with a small lower bound so recovery
does not freeze. A schema change or long gap resets history rather than mixing incompatible
states.

## Controlled fixture

[`controlled_motion.mp4`](../tests/assets/controlled_motion.mp4) is 640×360 at 30 FPS for
eight seconds. It contains translation, yaw transitions, scale, roll, neutral/smile
transitions, and a three-frame blank interval. It is generated entirely from the synthetic
pose guides. The exact phase table and regeneration command are in
[`tests/assets/README.md`](../tests/assets/README.md).

## Benchmark

The movement comparison processed frames 10–89 after a ten-frame warmup. Reference
enrollment was excluded. Each mode ran separately through the production MediaPipe tracker,
selector, renderer, and compositor on the same machine.

| Mode | Output FPS | Tracking mean / p95 | Processing mean / p95 | Total mean / p95 |
| --- | ---: | ---: | ---: | ---: |
| Smoothing on | 8.79 | 15.24 / 31.00 ms | 92.84 / 125.15 ms | 113.07 / 154.42 ms |
| Smoothing off | 9.41 | 17.84 / 32.00 ms | 83.03 / 108.01 ms | 105.32 / 140.50 ms |

Smoothing added about 9.8 ms mean processing time in this run and reduced throughput by
about 6.6%. Separate process runs and operating-system scheduling introduce normal timing
variation; these values are a development baseline rather than a guarantee.

The isolated dropout comparison measured 25 frames around the blank interval:

| Mode | Frames processed | No-face frames | Held frames |
| --- | ---: | ---: | ---: |
| Smoothing on | 25 / 25 | 0 | 3 |
| Smoothing off | 22 / 25 | 3 | 0 |

Run the same comparisons with:

```powershell
.\.venv\Scripts\python.exe tools\benchmarks\temporal_pipeline.py --max-frames 90 --warmup-frames 10
.\.venv\Scripts\python.exe tools\benchmarks\temporal_pipeline.py --max-frames 90 --warmup-frames 10 --disable-smoothing
.\.venv\Scripts\python.exe tools\benchmarks\temporal_pipeline.py --start-frame 210 --max-frames 30 --warmup-frames 5
```

## Remaining visual artifacts

- Large yaw changes can stretch cheek, nose, and eye texture because triangles reveal no
  previously hidden facial surface.
- A blink collapses the open-eye reference texture geometrically. It does not synthesize a
  true closed eyelid.
- Mouth opening uses warped lip landmarks and a dark cavity. Teeth, tongue, and changing
  inner-mouth detail are not reconstructed.
- Rapid expression changes can briefly soften or ghost between neutral and smile references.
- Lighting changes can still expose a face-oval boundary despite local color matching.
- The sparse initial library has no combined yaw-plus-pitch or side-smile references.
- The 150 ms hold intentionally freezes pose during a short loss; longer losses stop
  processed output.
- CPU geometric output remains substantially below the 30 FPS camera target.

## Deterministic-rendering evaluation

Deterministic rendering is sufficient for validating the pipeline architecture, reference
selection, temporal controls, frame identity, masking, failure behavior, and a visibly
functional prototype. It is not sufficient for the final visual-quality goal.

A later neural portrait renderer is justified for realistic eyelids, teeth and inner-mouth
detail, occluded surfaces at larger poses, and consistent texture under changing lighting.
It should replace the `FaceRenderer` implementation while keeping the current camera,
tracking, selection, stabilization, compositing, timing, and failure contracts.

