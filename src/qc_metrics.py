"""Simple, physically interpretable quality-control metrics.

These run before any machine learning and catch most faults on their own.
"""
from __future__ import annotations

import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
from skimage.registration import phase_cross_correlation
from skimage.restoration import estimate_sigma


def banding_score(img, axis=0, smooth=15):
    """Spread of column (axis=0) or row (axis=1) means after removing a smooth trend."""
    prof = img.mean(axis=axis)
    return float(np.std(prof - uniform_filter1d(prof, smooth, mode="nearest")))


def image_metrics(img) -> dict:
    lap = cv2.Laplacian(img.astype(np.float32), cv2.CV_32F)
    return dict(
        mean=float(img.mean()),
        saturated_frac=float((img >= 0.99).mean()),
        dark_frac=float((img <= 0.01).mean()),
        noise_sigma=float(estimate_sigma(img)),
        sharpness=float(lap.var()),
        banding_col=banding_score(img, axis=0),
        banding_row=banding_score(img, axis=1),
    )


def stuck_pixels(video, tol=1e-6) -> np.ndarray:
    """Pixels whose value never changes over time."""
    return video.std(axis=0) < tol


def frame_shifts(video) -> np.ndarray:
    """(T, 2) shift (dy, dx) of each frame relative to the previous one; row 0 is zero."""
    out = np.zeros((len(video), 2))
    for t in range(1, len(video)):
        out[t] = phase_cross_correlation(video[t - 1], video[t], upsample_factor=10, normalization=None)[0]
    return out


def register_video(video) -> tuple[np.ndarray, np.ndarray]:
    """Undo slow whole-field drift by accumulating frame-to-frame shifts.

    Returns (registered video, cumulative shifts). Edges are filled by reflection.
    """
    from scipy.ndimage import shift as nd_shift
    steps = frame_shifts(video)
    cum = np.cumsum(steps, axis=0)
    reg = np.stack([nd_shift(f, c, mode="reflect", order=1) for f, c in zip(video, cum)])
    return reg, cum


def video_metrics(video) -> pd.DataFrame:
    """Per-frame time series: brightness, shift magnitude, difference to previous frame."""
    diffs = np.r_[np.nan, np.abs(np.diff(video, axis=0)).mean(axis=(1, 2))]
    shifts = frame_shifts(video)
    return pd.DataFrame(dict(
        frame=np.arange(len(video)),
        brightness=video.mean(axis=(1, 2)),
        shift=np.hypot(*shifts.T),
        diff_prev=diffs,
        is_duplicate=diffs < 1e-9,
    ))
