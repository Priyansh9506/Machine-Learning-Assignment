"""Stage 8 baselines, written as scikit-learn classifiers so they are scored exactly like models.

1. MajorityBaseline: predicts the training late rate for every order ("always on time").
2. BusinessRule: flags an order when the seller's as-of late rate OR the distance exceeds a
   cut-off. Cut-offs are chosen on training data to maximise F1. This is what an operations
   team could do with a spreadsheet and no ML.
3. core_logreg(): logistic regression on five core features from the proposal.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from . import config as C

CORE_FEATURES = ["distance_km_max", "promised_days", "seller_late_rate_max", "freight_ratio", "purchase_month"]


class MajorityBaseline(ClassifierMixin, BaseEstimator):
    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        self.rate_ = float(np.mean(y))
        return self

    def predict_proba(self, X):
        p = np.full(len(X), self.rate_)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return np.zeros(len(X), dtype=int)


class BusinessRule(ClassifierMixin, BaseEstimator):
    """Flag if seller_late_rate_max > s or distance_km_max > d (grid-searched on training F1)."""

    def __init__(self, seller_col: str = "seller_late_rate_max", dist_col: str = "distance_km_max"):
        self.seller_col, self.dist_col = seller_col, dist_col

    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        s, d, y = X[self.seller_col].to_numpy(), X[self.dist_col].fillna(0).to_numpy(), np.asarray(y)
        s_grid = np.quantile(s, np.linspace(0.5, 0.99, 25))
        d_grid = np.append(np.quantile(d, np.linspace(0.5, 0.99, 25)), np.inf)
        best = (-1.0, None, None)
        for st in s_grid:
            for dt in d_grid:
                f1 = f1_score(y, (s > st) | (d > dt), zero_division=0)
                if f1 > best[0]:
                    best = (f1, st, dt)
        self.train_f1_, self.seller_cut_, self.dist_cut_ = best
        return self

    def predict(self, X):
        return ((X[self.seller_col].to_numpy() > self.seller_cut_)
                | (X[self.dist_col].fillna(0).to_numpy() > self.dist_cut_)).astype(int)

    def predict_proba(self, X):
        p = self.predict(X).astype(float)
        return np.column_stack([1 - p, p])


def core_logreg() -> Pipeline:
    """Plain L2 logistic regression on the five core features from the proposal."""
    num = CORE_FEATURES[:-1]
    pre = ColumnTransformer([
        ("num", Pipeline([("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                          ("scale", StandardScaler())]), num),
        ("month", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["purchase_month"]),
    ], verbose_feature_names_out=False)
    return Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=2000, random_state=C.SEED))])
