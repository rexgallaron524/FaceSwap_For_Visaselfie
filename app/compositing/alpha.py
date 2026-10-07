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
        if original.rgb.dtype != np.uint8 or face.rgb.dtype != np.uint8:
            raise StageError("Original and rendered RGB images must use uint8 pixels")
        if face.rgb.ndim != 3 or face.rgb.shape[2] != 3 or face.alpha.ndim != 2:
            raise StageError("Rendered face arrays have invalid dimensions")
        alpha = face.alpha
        if face.alpha.dtype != np.float32:
            raise StageError("Rendered alpha mask must contain float32 values")
        if face.active_region is None:
            if face.rgb.shape != original.rgb.shape or alpha.shape != original.rgb.shape[:2]:
                raise StageError("Rendered face dimensions do not match the original frame")
            if not np.isfinite(alpha).all() or np.any((alpha < 0.0) | (alpha > 1.0)):
                raise StageError("Rendered alpha mask values must be finite and in [0, 1]")
            active_y, active_x = np.nonzero(alpha > 1e-5)
            if len(active_x) == 0:
                output = np.ascontiguousarray(original.rgb.copy())
                output.setflags(write=False)
                return VideoFrame(original.frame_id, original.timestamp_ns, output)
            x0, x1 = int(active_x.min()), int(active_x.max()) + 1
            y0, y1 = int(active_y.min()), int(active_y.max()) + 1
        else:
            region = face.active_region
            if (
                len(region) != 4
                or any(type(value) is not int for value in region)
                or region[0] < 0
                or region[1] < 0
                or region[2] <= 0
                or region[3] <= 0
            ):
                raise StageError("Rendered face active region is invalid")
            x0, y0, region_width, region_height = region
            x1, y1 = x0 + region_width, y0 + region_height
            if x1 > original.rgb.shape[1] or y1 > original.rgb.shape[0]:
                raise StageError("Rendered face active region exceeds the output frame")
        region_shape = (y1 - y0, x1 - x0)
        if face.rgb.shape[:2] == region_shape and alpha.shape == region_shape:
            source = face.rgb
            local_alpha = alpha
        elif face.rgb.shape == original.rgb.shape and alpha.shape == original.rgb.shape[:2]:
            source = face.rgb[y0:y1, x0:x1]
            local_alpha = alpha[y0:y1, x0:x1]
        else:
            raise StageError("Rendered face dimensions do not match its active region")
        if not np.isfinite(local_alpha).all() or np.any((local_alpha < 0.0) | (local_alpha > 1.0)):
            raise StageError("Rendered alpha mask values must be finite and in [0, 1]")
        if not np.any(local_alpha > 1e-5):
            output = np.ascontiguousarray(original.rgb.copy())
            output.setflags(write=False)
            return VideoFrame(original.frame_id, original.timestamp_ns, output)
        target = original.rgb[y0:y1, x0:x1]
        statistics_mask = cv2.compare(local_alpha, 0.2, cv2.CMP_GE)
        if cv2.countNonZero(statistics_mask) > 1 and self._strength > 0.0:
            source_mean, source_stddev = cv2.meanStdDev(source, mask=statistics_mask)
            target_mean, target_stddev = cv2.meanStdDev(target, mask=statistics_mask)
            source_mean = source_mean.reshape(3).astype(np.float32)
            target_mean = target_mean.reshape(3).astype(np.float32)
            scale = (target_stddev.reshape(3) + 1.0) / (source_stddev.reshape(3) + 1.0)
            scale = np.clip(scale, 0.65, 1.5)
            gain = 1.0 - self._strength + scale * self._strength
            bias = (target_mean - source_mean * scale) * self._strength
            transform = np.zeros((3, 4), dtype=np.float32)
            transform[0, 0], transform[1, 1], transform[2, 2] = gain
            transform[:, 3] = bias
            source = cv2.transform(source, transform)

        composite = cv2.blendLinear(source, target, local_alpha, 1.0 - local_alpha)
        output = np.ascontiguousarray(original.rgb.copy())
        output[y0:y1, x0:x1] = composite
        output.setflags(write=False)
        return VideoFrame(original.frame_id, original.timestamp_ns, output)
