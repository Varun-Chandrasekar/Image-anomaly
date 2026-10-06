"""Anomaly detectors on the feature table.

Unsupervised: learn "normal" from clean training images only, then score everything.
Supervised: learn to name the artefact from labelled training images.
Evaluation is always on held-out source images.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM


def fit_unsupervised(X_clean, kind="iforest", seed=0):
    if kind == "iforest":
        model = IsolationForest(n_estimators=300, random_state=seed)
    elif kind == "ocsvm":
        model = OneClassSVM(nu=0.05, gamma="scale")
    else:
        raise ValueError(kind)
    return make_pipeline(StandardScaler(), model).fit(X_clean)


def anomaly_score(model, X):
    """Higher = more anomalous."""
    return -model.score_samples(X)


def fit_supervised(X, y, seed=0):
    return make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=300, random_state=seed)).fit(X, y)


def auc_by_group(table: pd.DataFrame, scores, by=("artefact", "level")) -> pd.DataFrame:
    """ROC-AUC of each artefact/level against the clean images of the same split."""
    clean = scores[table.artefact.values == "clean"]
    rows = []
    for key, idx in table[table.artefact != "clean"].groupby(list(by)).groups.items():
        key = key if isinstance(key, tuple) else (key,)
        s = scores[np.asarray(idx)]
        y = np.r_[np.zeros(len(clean)), np.ones(len(s))]
        rows.append(dict(zip(by, key), auc=roc_auc_score(y, np.r_[clean, s]), n=len(s)))
    return pd.DataFrame(rows)
