"""Wavelet and Fourier features.

Fourier tells you which frequencies are present but not where; a wavelet transform splits
the image into scales and three orientations (H, V, D) while keeping location.
The energy share of each sub-band is a compact fingerprint of the image.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pywt

from .qc_metrics import image_metrics


def wavelet_energies(img, wavelet="db2", level=3) -> dict:
    """Share of total energy in each sub-band. Level 1 = coarsest, `level` = finest."""
    coeffs = pywt.wavedec2(img, wavelet, level=level, mode="periodization")
    feats = {"approx": float(np.sum(coeffs[0] ** 2))}
    for lvl, (ch, cv, cd) in enumerate(coeffs[1:], start=1):
        feats[f"H{lvl}"] = float(np.sum(ch ** 2))
        feats[f"V{lvl}"] = float(np.sum(cv ** 2))
        feats[f"D{lvl}"] = float(np.sum(cd ** 2))
    total = sum(feats.values())
    return {f"wav_{k}": v / total for k, v in feats.items()}


def radial_spectrum(img, n_bins=32) -> np.ndarray:
    """Mean log power in rings of increasing spatial frequency (0 = DC, last = Nyquist)."""
    f = np.fft.fftshift(np.fft.fft2(img - img.mean()))
    p = np.log1p(np.abs(f) ** 2)
    h, w = img.shape
    yy, xx = np.mgrid[:h, :w]
    r = np.hypot(yy - h / 2, xx - w / 2) / (min(h, w) / 2)
    bins = np.minimum((r * n_bins).astype(int), n_bins)
    return np.array([p[bins == b].mean() for b in range(n_bins)])


def blockiness(img, block=8) -> float:
    """JPEG 8x8 blockiness: jump across block boundaries vs jump inside blocks."""
    d = np.abs(np.diff(img, axis=1))
    on = d[:, block - 1::block].mean()
    off = np.delete(d, np.s_[block - 1::block], axis=1).mean()
    return float(on / (off + 1e-12))


def wavelet_denoise(img, wavelet="db2", level=3, k=3.0):
    """Soft-threshold the detail coefficients (universal-style threshold) and reconstruct."""
    coeffs = pywt.wavedec2(img, wavelet, level=level, mode="periodization")
    sigma = np.median(np.abs(coeffs[-1][2])) / 0.6745  # noise estimate from finest diagonal
    t = k * sigma
    new = [coeffs[0]] + [tuple(pywt.threshold(c, t, mode="soft") for c in lvl) for lvl in coeffs[1:]]
    return np.clip(pywt.waverec2(new, wavelet, mode="periodization"), 0, 1)


def feature_vector(img) -> dict:
    return {**image_metrics(img), **wavelet_energies(img), "blockiness": blockiness(img)}


def feature_table(images) -> pd.DataFrame:
    return pd.DataFrame([feature_vector(im) for im in images])
