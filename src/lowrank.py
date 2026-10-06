"""Low-rank and tensor decompositions for video.

Reshape the video into a pixels x frames matrix. A static background is nearly the same
in every column, so it is low-rank; moving objects and faults are sparse.
Robust PCA splits M = L (low rank) + S (sparse).
"""
from __future__ import annotations

import numpy as np


def to_matrix(video):
    t, h, w = video.shape
    return video.reshape(t, h * w).T  # pixels x frames


def from_matrix(m, shape):
    return m.T.reshape(shape)


def singular_values(video):
    return np.linalg.svd(to_matrix(video), compute_uv=False)


def rank_curve(video, max_rank=20):
    """Relative reconstruction error of the best rank-k approximation, k = 1..max_rank."""
    s = singular_values(video)
    total = np.sum(s ** 2)
    return np.array([np.sqrt(np.sum(s[k:] ** 2) / total) for k in range(1, max_rank + 1)])


def rpca(M, lam=None, tol=1e-7, max_iter=500):
    """Robust PCA by the inexact augmented Lagrange multiplier method.

    Solves min ||L||_* + lam * ||S||_1  subject to  M = L + S.
    Returns (L, S, n_iter).
    """
    m, n = M.shape
    lam = lam or 1 / np.sqrt(max(m, n))
    norm2 = np.linalg.norm(M, 2)
    mu = 1.25 / norm2
    Y = M / max(norm2, np.abs(M).max() / lam)
    L = np.zeros_like(M)
    S = np.zeros_like(M)
    normM = np.linalg.norm(M)
    for it in range(1, max_iter + 1):
        U, sig, Vt = np.linalg.svd(M - S + Y / mu, full_matrices=False)
        L = (U * np.maximum(sig - 1 / mu, 0)) @ Vt
        R = M - L + Y / mu
        S = np.sign(R) * np.maximum(np.abs(R) - lam / mu, 0)
        Z = M - L - S
        Y += mu * Z
        mu = min(mu * 1.5, 1e7)
        if np.linalg.norm(Z) / normM < tol:
            break
    return L, S, it


def video_rpca(video, **kw):
    """RPCA on a (T, H, W) video. Returns background L and foreground S as videos."""
    L, S, _ = rpca(to_matrix(video), **kw)
    return from_matrix(L, video.shape), from_matrix(S, video.shape)


def frame_scores(video, L, S) -> dict:
    """Per-frame anomaly scores: energy of the sparse part and change of the background."""
    s_energy = np.sqrt((S ** 2).mean(axis=(1, 2)))
    l_mean = L.mean(axis=(1, 2))
    return dict(sparse_energy=s_energy,
                background_jump=np.r_[0, np.abs(np.diff(l_mean))])


def cp_time_factors(video, rank=5, seed=0):
    """CP (PARAFAC) decomposition of the (H, W, T) tensor; returns the T x rank time factor."""
    import tensorly as tl
    from tensorly.decomposition import parafac
    cp = parafac(tl.tensor(np.moveaxis(video, 0, -1)), rank=rank, init="random", random_state=seed, n_iter_max=200)
    return np.asarray(cp.factors[2]) * np.asarray(cp.weights)[None, :]
