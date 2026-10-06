import numpy as np

from src.vvuq import bootstrap_auc_ci, conformal_threshold


def test_conformal_false_alarm_rate():
    rng = np.random.default_rng(0)
    alpha = 0.1
    rates = []
    for _ in range(200):
        cal, new = rng.normal(size=100), rng.normal(size=1000)
        rates.append(np.mean(new > conformal_threshold(cal, alpha)))
    assert np.mean(rates) <= alpha + 0.01


def test_bootstrap_ci_contains_point():
    rng = np.random.default_rng(0)
    y = np.r_[np.zeros(100), np.ones(100)]
    s = np.r_[rng.normal(0, 1, 100), rng.normal(1, 1, 100)]
    auc, lo, hi = bootstrap_auc_ci(y, s, n_boot=300)
    assert lo < auc < hi
