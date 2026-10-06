import numpy as np

from src.lowrank import rank_curve, rpca


def test_rpca_recovers_known_low_rank_plus_sparse():
    """Verification: a synthetic problem with a known answer."""
    rng = np.random.default_rng(0)
    m, n, r = 200, 60, 2
    L0 = rng.standard_normal((m, r)) @ rng.standard_normal((r, n))
    S0 = np.zeros((m, n))
    mask = rng.random((m, n)) < 0.05
    S0[mask] = rng.choice([-10, 10], mask.sum())
    L, S, _ = rpca(L0 + S0)
    assert np.linalg.norm(L - L0) / np.linalg.norm(L0) < 1e-3
    assert np.linalg.norm(S - S0) / np.linalg.norm(S0) < 1e-3


def test_rank_curve_decreases():
    rng = np.random.default_rng(0)
    v = rng.random((30, 16, 16))
    assert np.all(np.diff(rank_curve(v, 10)) <= 1e-12)
