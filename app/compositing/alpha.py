"""Masked face compositing with lightweight local color correction."""

from __future__ import annotations

import cv2
import numpy as np

from app.pipeline.types import RenderedFace, StageError, VideoFrame


class AlphaFaceCompositor:
    """Feather and color-match a rendered face while preserving the source frame."""

    def __init__(self, *, color_match_strength: float = 0.65) -> None:
        if not 0.0 <= color_match_strength <= 1.0:
            raise ValueError("Color-match strength must be in [0, 1]")
        self._strength = color_match_strength

    def composite(self, original: VideoFrame, face: RenderedFace) -> VideoFrame:
        if (original.frame_id, original.timestamp_ns) != (face.frame_id, face.timestamp_ns):
            raise StageError("Original and rendered face identities do not match")
        if original.rgb.ndim != 3 or original.rgb.shape[2] != 3:
            raise StageError("Original frame must be an H x W x 3 RGB image")
        if face.rgb.shape != original.rgb.shape or face.alpha.shape != original.rgb.shape[:2]:
            raise StageError("Rendered face dimensions do not match the original frame")
        if original.rgb.dtype != np.uint8 or face.rgb.dtype != np.uint8:
            raise StageError("Original and rendered RGB images must use uint8 pixels")
        if face.alpha.dtype != np.float32 or not np.isfinite(face.alpha).all():
            raise StageError("Rendered alpha mask must contain finite float32 values")
        if np.any(face.alpha < 0.0) or np.any(face.alpha > 1.0):
            raise StageError("Rendered alpha mask values must be in [0, 1]")

        alpha = face.alpha
        active_y, active_x = np.nonzero(alpha > 1e-5)
        if len(active_x) == 0:
            output = np.ascontiguousarray(original.rgb.copy())
            output.setflags(write=False)
            return VideoFrame(original.frame_id, original.timestamp_ns, output)

        x0, x1 = int(active_x.min()), int(active_x.max()) + 1
        y0, y1 = int(active_y.min()), int(active_y.max()) + 1
        local_alpha = alpha[y0:y1, x0:x1]
        source = face.rgb[y0:y1, x0:x1].astype(np.float32)
        target = original.rgb[y0:y1, x0:x1].astype(np.float32)
        statistics_mask = np.where(local_alpha >= 0.2, 255, 0).astype(np.uint8)
        if cv2.countNonZero(statistics_mask) > 1 and self._strength > 0.0:
            source_mean, source_stddev = cv2.meanStdDev(
                face.rgb[y0:y1, x0:x1], mask=statistics_mask
            )
            target_mean, target_stddev = cv2.meanStdDev(
                original.rgb[y0:y1, x0:x1], mask=statistics_mask
            )
            source_mean = source_mean.reshape(3).astype(np.float32)
            target_mean = target_mean.reshape(3).astype(np.float32)
            scale = (target_stddev.reshape(3) + 1.0) / (source_stddev.reshape(3) + 1.0)
            scale = np.clip(scale, 0.65, 1.5)
            matched = (source - source_mean) * scale + target_mean
            source = source * (1.0 - self._strength) + matched * self._strength

        # Suppress single-pixel mesh seams without spreading beyond the supplied alpha.
        source = cv2.GaussianBlur(source, (3, 3), 0.0)
        opacity = local_alpha[..., None]
        composite = np.clip(np.rint(source * opacity + target * (1.0 - opacity)), 0, 255).astype(
            np.uint8
        )
        output = np.ascontiguousarray(original.rgb.copy())
        output[y0:y1, x0:x1] = composite
        output.setflags(write=False)
        return VideoFrame(original.frame_id, original.timestamp_ns, output)
