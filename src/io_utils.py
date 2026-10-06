"""Loading images and videos. Everything is returned as float64 in [0, 1]."""
from __future__ import annotations

import cv2
import numpy as np
from skimage import color, data, transform, util

# Grayscale-friendly stills shipped with scikit-image (no download needed).
STILLS = ["camera", "coins", "moon", "text", "brick", "grass", "gravel", "astronaut"]


def load_still(name: str, size: int = 256) -> np.ndarray:
    """Load a scikit-image sample image as a size x size grayscale float image."""
    img = util.img_as_float(getattr(data, name)())
    if img.ndim == 3:
        img = color.rgb2gray(img[..., :3])
    return transform.resize(img, (size, size), anti_aliasing=True)


def load_stills(names=STILLS, size: int = 256) -> dict[str, np.ndarray]:
    return {n: load_still(n, size) for n in names}


def read_video(path: str, channel: str = "gray", scale: float | None = None,
               max_frames: int | None = None) -> np.ndarray:
    """Read a video file into a (T, H, W) float array in [0, 1].

    channel: "gray", or one of "b", "g", "r" to keep a single colour channel
    (the ferrofluid video carries most of its signal in red).
    scale: optional resize factor, e.g. 0.25 turns 1024x768 into 256x192.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(path)
    frames = []
    while max_frames is None or len(frames) < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if channel == "gray":
            f = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            f = frame[..., "bgr".index(channel)]
        if scale is not None:
            f = cv2.resize(f, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        frames.append(f)
    cap.release()
    return np.stack(frames).astype(np.float64) / 255.0


def write_video(path: str, video: np.ndarray, fps: float = 10.0) -> None:
    """Write a (T, H, W) float video to an AVI (MJPG) file for viewing."""
    t, h, w = video.shape
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h), isColor=False)
    for f in video:
        out.write(np.clip(f * 255 + 0.5, 0, 255).astype(np.uint8))
    out.release()
