"""Generate the deterministic Milestone 6 face-motion benchmark clip."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

FPS = 30
DURATION_SECONDS = 8
FRAME_SIZE = (640, 360)


def _load_guide(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read pose guide: {path}")
    height, width = image.shape[:2]
    target_ratio = FRAME_SIZE[0] / FRAME_SIZE[1]
    crop_height = min(height, round(width / target_ratio))
    y0 = (height - crop_height) // 2
    cropped = image[y0 : y0 + crop_height]
    return cv2.resize(cropped, FRAME_SIZE, interpolation=cv2.INTER_AREA)


def _blend(left: np.ndarray, right: np.ndarray, amount: float) -> np.ndarray:
    return cv2.addWeighted(left, 1.0 - amount, right, amount, 0.0)


def _triangle_wave(value: float) -> float:
    return 1.0 - abs(2.0 * value - 1.0)


def build_frame(guides: dict[str, np.ndarray], seconds: float) -> np.ndarray:
    image = guides["front-neutral"]
    translation_x = 0.0
    scale = 1.0
    roll = 0.0

    if 1.0 <= seconds < 2.5:
        phase = (seconds - 1.0) / 1.5
        translation_x = -55.0 + 110.0 * phase
    elif 2.5 <= seconds < 4.0:
        phase = (seconds - 2.5) / 1.5
        if phase < 0.25:
            image = _blend(guides["front-neutral"], guides["left-20"], phase / 0.25)
        elif phase < 0.5:
            image = _blend(guides["left-20"], guides["front-neutral"], (phase - 0.25) / 0.25)
        elif phase < 0.75:
            image = _blend(guides["front-neutral"], guides["right-20"], (phase - 0.5) / 0.25)
        else:
            image = _blend(guides["right-20"], guides["front-neutral"], (phase - 0.75) / 0.25)
    elif 4.0 <= seconds < 5.0:
        scale = 0.88 + 0.24 * _triangle_wave(seconds - 4.0)
    elif 5.0 <= seconds < 6.0:
        phase = seconds - 5.0
        roll = -9.0 + 18.0 * _triangle_wave(phase)
    elif 6.0 <= seconds < 7.0:
        phase = _triangle_wave(seconds - 6.0)
        image = _blend(guides["front-neutral"], guides["front-smile"], phase)

    transform = cv2.getRotationMatrix2D((FRAME_SIZE[0] / 2, FRAME_SIZE[1] / 2), roll, scale)
    transform[0, 2] += translation_x
    frame = cv2.warpAffine(
        image,
        transform,
        FRAME_SIZE,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    # Three blank frames create a repeatable 100 ms tracking dropout near the end.
    if 7.40 <= seconds < 7.50:
        frame[:] = (205, 208, 212)
    return frame


def main() -> int:
    repository = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=repository / "tests" / "assets" / "controlled_motion.mp4",
    )
    args = parser.parse_args()

    guide_root = repository / "assets" / "pose_guides"
    guides = {
        name: _load_guide(guide_root / f"{name}.png")
        for name in ("front-neutral", "front-smile", "left-20", "right-20")
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.output),
        cv2.VideoWriter.fourcc(*"mp4v"),
        FPS,
        FRAME_SIZE,
    )
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not create an MP4 writer")
    try:
        for frame_index in range(FPS * DURATION_SECONDS):
            writer.write(build_frame(guides, frame_index / FPS))
    finally:
        writer.release()
    print(f"Wrote {FPS * DURATION_SECONDS} frames to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
