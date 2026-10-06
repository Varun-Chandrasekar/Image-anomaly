"""Verification, validation and uncertainty quantification helpers."""
from __future__ import annotations

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, roc_auc_score


def conformal_threshold(cal_scores, alpha=0.05):
    """Split-conformal threshold on clean calibration scores.

    Flag a new sample when its score exceeds the threshold. If the calibration data and
    new clean data are exchangeable, the false-alarm rate is at most alpha.
    """
    s = np.sort(np.asarray(cal_scores))
    n = len(s)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return np.inf if k > n else float(s[k - 1])


def bootstrap_auc_ci(y, scores, n_boot=1000, ci=0.95, seed=0):
    """Point AUC and percentile bootstrap confidence interval."""
    rng = np.random.default_rng(seed)
    y, scores = np.asarray(y), np.asarray(scores)
    aucs = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        aucs.append(roc_auc_score(y[i], scores[i]))
    lo, hi = np.quantile(aucs, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    return float(roc_auc_score(y, scores)), float(lo), float(hi)


def reliability(y, prob, n_bins=10):
    """Calibration curve and Brier score for predicted probabilities."""
    frac_pos, mean_pred = calibration_curve(y, prob, n_bins=n_bins, strategy="quantile")
    return dict(frac_pos=frac_pos, mean_pred=mean_pred, brier=float(brier_score_loss(y, prob)))
