# Model assets

## Face Landmarker

`face_landmarker.task` is the official MediaPipe Face Landmarker float16 model bundle.
It is used locally in image mode to validate and preprocess enrollment references.
It contains face detection, 478-point face mesh, and blendshape models; it is not a
portrait-generation model and is never sent over the network.

- Source: `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task`
- Retrieved: 2026-10-07
- Size: 3,758,596 bytes
- SHA-256: `64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff`
- License: Apache License 2.0; see `LICENSE.face-landmarker.txt`
- Documentation and model cards: <https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker/index#models>
- Component model cards: [Face Mesh](https://storage.googleapis.com/mediapipe-assets/Model%20Card%20MediaPipe%20Face%20Mesh%20V2.pdf), [Blendshape](https://storage.googleapis.com/mediapipe-assets/Model%20Card%20Blendshape%20V2.pdf), and [BlazeFace](https://storage.googleapis.com/mediapipe-assets/MediaPipe%20BlazeFace%20Model%20Card%20%28Short%20Range%29.pdf)

The model is redistributed with the project and copied into built wheels as
`app/reference/assets/face_landmarker.task`, so enrollment works offline after installation.
Verify the checksum when updating it. Neural portrait refinement remains deferred until
deterministic face geometry is evaluated.
