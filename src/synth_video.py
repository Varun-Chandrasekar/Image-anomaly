"""Synthetic videos with exact per-frame ground truth.

Two kinds of anomaly are kept apart on purpose:
  * events  - real things happening in the scene (a moving blob / aggregate)
  * faults  - sensor or pipeline problems (gain jump, jitter, duplicate frame, stuck pixels)
A credible detector should flag faults without confusing them with events, and vice versa.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from skimage import restoration, transform

FAULTS = ("none", "gain", "jitter", "duplicate", "drop")


def _gaussian_blob(shape, cy, cx, sy, sx, theta=0.0):
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    y, x = yy - cy, xx - cx
    c, s = np.cos(theta), np.sin(theta)
    u, v = c * x + s * y, -s * x + c * y
    return np.exp(-0.5 * ((u / sx) ** 2 + (v / sy) ** 2))


def inject_faults(video, rng, gain=(80, 91, 1.3), jitter=(120, 2), duplicate=150,
                  n_stuck=20, drop=None, drift=0.0):
    """Inject sensor faults into a (T, H, W) video. Returns (video, fault_per_frame, stuck_mask).

    gain:      (start, stop, factor) frames start..stop-1 are multiplied by factor
    jitter:    (frame, pixels)       that frame is shifted sideways
    duplicate: frame index           that frame is an exact copy of the previous one
    drop:      frame index or None   that frame is removed (the clip gets one frame shorter)
    drift:     total brightness change spread linearly over the clip (slow drift)
    n_stuck:   number of pixels frozen at 0 or 1 for the whole clip
    """
    v = video.copy()
    t, h, w = v.shape
    fault = np.array(["none"] * t, dtype=object)
    if drift:
        v += np.linspace(0, drift, t)[:, None, None]
    if gain:
        a, b, g = gain
        v[a:b] *= g
        fault[a:b] = "gain"
    if jitter:
        f, px = jitter
        v[f] = np.roll(v[f], px, axis=1)
        fault[f] = "jitter"
    if duplicate is not None:
        v[duplicate] = v[duplicate - 1]
        fault[duplicate] = "duplicate"
    stuck = np.zeros((h, w), bool)
    idx = rng.choice(h * w, n_stuck, replace=False)
    stuck.flat[idx] = True
    v[:, stuck] = rng.choice([0.0, 1.0], n_stuck)
    v = np.clip(v, 0, 1)
    if drop is not None:
        v = np.delete(v, drop, axis=0)
        fault = np.delete(fault, drop)
        fault[drop] = "drop"  # the frame after the gap
    return v, fault, stuck


def make_blob_video(background, n_frames=200, size=128, n_blobs=2, noise=0.01, seed=0, faults=True):
    """Fixed background + small bright blobs moving across it + sensor noise + faults.

    Returns (video, labels, meta). labels has one row per frame:
    frame, has_event, fault. meta holds the stuck-pixel mask and blob tracks.
    """
    rng = np.random.default_rng(seed)
    bg = transform.resize(background, (size, size), anti_aliasing=True)
    video = np.repeat(bg[None], n_frames, axis=0)
    tracks = np.full((n_blobs, n_frames, 2), np.nan)
    has_event = np.zeros(n_frames, bool)
    for b in range(n_blobs):
        start = rng.integers(10, n_frames // 2)
        dur = rng.integers(40, 80)
        y0, y1 = rng.uniform(15, size - 15, 2)
        for i, t in enumerate(range(start, min(start + dur, n_frames))):
            frac = i / dur
            cy, cx = y0 + (y1 - y0) * frac, 5 + (size - 10) * frac
            video[t] += 0.6 * _gaussian_blob((size, size), cy, cx, 2.5, 2.5)
            tracks[b, t] = cy, cx
            has_event[t] = True
    video = np.clip(video + rng.normal(0, noise, video.shape), 0, 1)
    fault = np.array(["none"] * n_frames, dtype=object)
    stuck = np.zeros((size, size), bool)
    if faults:
        video, fault, stuck = inject_faults(video, rng)
    labels = pd.DataFrame(dict(frame=np.arange(len(video)), has_event=has_event[: len(video)], fault=fault))
    return video, labels, dict(stuck_mask=stuck, tracks=tracks)


def make_lc_video(real_video, n_frames=200, n_aggregates=2, seed=0, faults=True):
    """Liquid-crystal-like video built from a real (registered) microscope clip.

    Background = per-pixel median of the real frames. Dark elliptical "aggregates" move
    with Brownian motion plus drift. Noise is matched to the real clip's noise level.
    """
    rng = np.random.default_rng(seed)
    bg = np.median(real_video, axis=0)
    h, w = bg.shape
    noise = float(np.median([restoration.estimate_sigma(f) for f in real_video[:: max(1, len(real_video) // 10)]]))
    video = np.repeat(bg[None], n_frames, axis=0)
    tracks = np.zeros((n_aggregates, n_frames, 2))
    for a in range(n_aggregates):
        pos = rng.uniform([0.2 * h, 0.2 * w], [0.8 * h, 0.8 * w])
        vel = rng.normal(0, 0.3, 2)
        sy, sx = rng.uniform(2, 5, 2) * h / 192
        theta = rng.uniform(0, np.pi)
        for t in range(n_frames):
            pos = np.clip(pos + vel + rng.normal(0, 0.5, 2), [5, 5], [h - 5, w - 5])
            theta += rng.normal(0, 0.05)
            video[t] *= 1 - 0.7 * _gaussian_blob((h, w), *pos, sy, sx, theta)
            tracks[a, t] = pos
    video = np.clip(video + rng.normal(0, noise, video.shape), 0, 1)
    fault = np.array(["none"] * n_frames, dtype=object)
    stuck = np.zeros((h, w), bool)
    if faults:
        video, fault, stuck = inject_faults(video, rng)
    labels = pd.DataFrame(dict(frame=np.arange(len(video)), has_event=True, fault=fault))
    return video, labels, dict(stuck_mask=stuck, tracks=tracks, noise_sigma=noise)
