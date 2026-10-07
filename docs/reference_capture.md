# Reference capture plan

This is enrollment guidance for a later milestone. No capture, preprocessing,
reference file format, or library loader is implemented yet.

Prepare references only for the user or an explicitly consenting subject, using
consistent lighting, camera distance, and framing. Keep the complete facial region
visible. The background will not be used for replacement.

Initial poses:

1. Front, neutral expression.
2. Front, smiling.
3. Subject turns left approximately 20 degrees.
4. Subject turns left approximately 40 degrees.
5. Subject turns right approximately 20 degrees.
6. Subject turns right approximately 40 degrees.
7. Looking slightly up.
8. Looking slightly down.

Left/right above refer to the subject. For unmirrored images, subject-left corresponds
to positive yaw in the internal convention. Pose measurements, not filename labels,
will drive selection. Allow additional poses and individual reference replacement.
Roll and distance changes should be handled geometrically.

Future enrollment must validate each image, find/align one face, normalize it, and
cache the processed image/landmarks. Reject ambiguous/multiple-face inputs. Preprocess
on load or reference change rather than during live frame processing. Define a
versioned manifest and cache invalidation strategy in the reference-library milestone.

Keep personal references in the ignored `references/` directory or outside the repo.
Do not commit face images or recordings without explicit consent and a clear fixture
license/provenance record. Synthetic fixtures should be preferred for deterministic tests.
