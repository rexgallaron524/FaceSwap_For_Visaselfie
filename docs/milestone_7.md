# Milestone 7 — Neural Refinement Evaluation

## Decision

The Milestones 5–6 renderer remains useful as a deterministic baseline, diagnostics tool,
and fallback. It does not meet the intended visual-quality requirement across pose and
expression changes. A neural portrait-animation stage is justified, but adding model
weights to the default application is not justified yet.

This milestone adds a contained integration proof of concept: `NeuralFaceRenderer` implements
the existing `FaceRenderer` protocol, caches reference appearance state, preserves continuous
reference weights, validates outputs, and profiles cold preparation and warm inference. It
loads an optional local provider only when selected. No neural dependency or weight file is
downloaded or committed.

## Current quality compared with requirements

| Requirement | Geometric result | Remaining problem | Neural opportunity |
| --- | --- | --- | --- |
| Front pose and small motion | Functional with a clear enrolled image | Mesh and mask edges can remain visible under lighting changes | Learned synthesis can harmonize texture and local shading |
| Large yaw | Reference selection is continuous | Affine triangles stretch the nose, eye, and far cheek; unseen texture is unavailable | A learned 3D-aware warp can infer partially occluded surfaces |
| Profile rendering | Side references reduce the pose gap | The 2D mesh cannot create a credible silhouette or reconstruct hidden facial regions | Implicit keypoints and learned decoding are better suited to viewpoint changes |
| Mouth activity | Lip landmarks open and close; a cavity is inserted | No teeth, tongue, lip interior, or view-dependent mouth detail | Lip retargeting plus a learned decoder can synthesize inner-mouth appearance |
| Smile/blink | Smoothed blendshapes move landmarks | Eyelids and cheeks are texture warps; smile volume and nasolabial changes are absent | Expression retargeting can generate appearance changes around eyes, mouth, and cheeks |
| Cheek deformation | Landmarks move the local triangular mesh | Triangle shear can look rubbery and preserve incorrect shading | Learned deformation can model nonlinear cheek motion |
| Reference interpolation | Weights change continuously | Pixel blending can ghost double features when pose/expression references disagree | Appearance features can be cached and combined or the closest source can be animated continuously |
| Temporal stability | Milestone 6 removes most abrupt changes | Fast motion can still blur or briefly freeze during a tracking hold | A neural runtime still needs the existing stabilizer and explicit drop policy |
| Body/background preservation | Facial alpha mask preserves the original frame | Face-oval seams remain possible | Neural output must still use the current compositor and facial-only mask contract |

The controlled 640×360 benchmark on this development system measured 8.34 processed FPS
with smoothing, 19.64 ms mean tracking latency, 96.58 ms mean post-tracking processing, and
119.19 ms mean total latency. Raw temporal mode measured 8.96 FPS and 110.59 ms mean total
latency. These values include geometric rendering and compositing and show that the current
CPU implementation is already below the 30 FPS target.

## LivePortrait assessment

[LivePortrait](https://github.com/KlingAIResearch/LivePortrait) is an appropriate reference
because it separates source appearance extraction from per-frame motion extraction,
keypoint transformation, warping, decoding, stitching, and eye/lip retargeting. That split
matches the requirement to preprocess enrolled reference appearance once.

The project reports 12.8 ms model inference on an RTX 4090 with PyTorch. Its published module
table totals about 130 million parameters and about 500 MB across appearance, motion,
warping, generator, and retargeting modules. That number excludes application tracking,
cropping, reference selection, compositing, transfers, and UI delivery, so it is not an
end-to-end FaceLive latency claim.

Important integration gaps remain:

- LivePortrait normally derives motion from a driving RGB crop and its own implicit
  keypoints. FaceLive's `FaceRenderer` receives a backend-neutral `FaceState`, not the raw
  driving frame. A production provider must map the stabilized pose, eye, lip, and expression
  signals into LivePortrait motion state or introduce a separate, explicit driving-crop input
  without leaking model types into the UI.
- The official setup targets a separate Python 3.10 environment and pins NumPy 1.26.4 and
  OpenCV 4.10. FaceLive uses Python 3.12, NumPy 2.2+, and OpenCV 5. Installing it directly into
  the application environment would create dependency conflicts.
- The development system used for this evaluation exposes no NVIDIA runtime and has no
  PyTorch or ONNX Runtime installed. A meaningful local model benchmark was therefore not
  possible without downloading a large stack and weights.

The production recommendation is a local model worker in its own pinned environment. A small
provider module can speak to that worker and implement the runtime contract used by
`NeuralFaceRenderer`. This keeps model imports, CUDA lifetime, and version conflicts outside
the PySide6 process while retaining local inference.

## Contained adapter proof of concept

`app/rendering/neural.py` defines:

- `PortraitAnimationRuntime`, the model-specific preparation/inference boundary;
- `PythonModulePortraitRuntime`, an optional local provider loader;
- `NeuralFaceRenderer`, the existing pipeline adapter;
- `WeightedAppearance`, preserving pose/expression library weights; and
- `NeuralRendererMetrics`, measuring preparation and inference separately.

The adapter prepares every library reference once on first use. It reuses the opaque
appearance object while the `PreparedReference` is unchanged, releases removed/replaced
entries, and clears device state on close. The runtime must return the same full-frame RGB
and alpha contract as the geometric renderer, including matching frame identity.

The contract-only benchmark at 1280×720 measures cache and full-frame validation/copy cost;
it intentionally excludes model inference. Run it with:

```powershell
.\.venv\Scripts\python.exe tools\benchmarks\neural_adapter.py
```

Observed on the development system with eight 256×256 references and 200 warm frames:

| Measurement | Result |
| --- | ---: |
| Cold adapter call | 7.334 ms |
| One-time cache preparation with null provider | 1.383 ms |
| Warm adapter total mean / p95 | 3.997 / 5.178 ms |
| Null runtime call mean / p95 | 0.002 / 0.004 ms |
| Warm cache misses | 0 |

The warm total is dominated by validating and taking ownership of the required 1280×720 RGB
and alpha outputs. The null runtime number is a contract baseline, not neural inference.

The real provider's `render` call is timed by the adapter, so the same `inference_ms` metric
will profile local inference once a licensed runtime and weights are installed.

## Backend configuration

Geometric rendering remains the default:

```toml
[renderer]
backend = "geometric"
```

An optional provider can be selected explicitly:

```toml
[renderer]
backend = "liveportrait"
runtime_module = "facelive_liveportrait"
model_directory = "../models/liveportrait"
device = "cuda"
```

The provider module must export:

```python
def create_runtime(*, model_directory: Path, device: str) -> PortraitAnimationRuntime: ...
```

If it is absent, camera and tracking can continue while processed rendering reports an
actionable unavailable state. There is no automatic download or fallback to raw video.

## Licensing and distribution

- The official LivePortrait source repository is MIT licensed.
- Its license explicitly states that bundled InsightFace models are restricted to
  non-commercial research and must be removed/replaced for commercial use.
- FaceLive already has MediaPipe tracking and aligned enrolled references. A provider should
  bypass or replace InsightFace detection rather than redistribute those models.
- Model checkpoints, every transitive dependency, and any separately distributed provider
  require a release-specific license inventory before packaging.
- Model files should remain external, versioned, hash-verified local assets. Their provenance
  and accepted license should be recorded alongside the files.

Sources reviewed on 2026-10-07: the official
[repository](https://github.com/KlingAIResearch/LivePortrait),
[license](https://github.com/KlingAIResearch/LivePortrait/blob/main/LICENSE),
[speed table](https://github.com/KlingAIResearch/LivePortrait/blob/main/assets/docs/speed.md),
and [paper](https://arxiv.org/abs/2407.03168). The source API was inspected at commit
`9b294b3d0536135442ea73cb01e6cb3ca7029dd3`.

## Stable decisions for later milestones

- Keep `FaceRenderer` as the only pipeline-facing render boundary.
- Keep full-frame RGB plus facial alpha output so the existing compositor owns body and
  background preservation.
- Cache reference appearance independently from per-frame motion.
- Keep the geometric renderer available for fallback, comparison, and diagnostics.
- Keep model packages and weights optional and local; do not import them from UI or future
  virtual-camera code.
- Do not call a model service over the network.
- Profile preparation, inference, compositing, and complete-frame latency separately.

