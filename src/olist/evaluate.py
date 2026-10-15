"""Evaluation helpers: threshold metrics, per-period scores, bootstrap intervals, top-k."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, brier_score_loss,
                             f1_score, precision_recall_curve, precision_score, recall_score,
                             roc_auc_score)

from . import config as C


def scores(y, p, threshold: float = 0.5) -> dict:
    """Ranking metrics (threshold-free) plus classification metrics at `threshold`."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    yhat = (p >= threshold).astype(int)
    return {
        "pr_auc": average_precision_score(y, p),
        "roc_auc": roc_auc_score(y, p),
        "brier": brier_score_loss(y, np.clip(p, 0, 1)),
        "precision": precision_score(y, yhat, zero_division=0),
        "recall": recall_score(y, yhat, zero_division=0),
        "f1": f1_score(y, yhat, zero_division=0),
        "balanced_acc": balanced_accuracy_score(y, yhat),
        "flagged_%": yhat.mean() * 100,
        "base_rate_%": y.mean() * 100,
    }


def best_f1_threshold(y, p) -> float:
    """Threshold that maximises F1 on (y, p). Choose it on validation, never on test."""
    prec, rec, thr = precision_recall_curve(np.asarray(y).astype(int), p)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    return float(thr[np.nanargmax(f1[:-1])])


def per_period(y, p, period) -> pd.DataFrame:
    """PR-AUC, ROC-AUC and base rate for each period label (e.g. purchase month)."""
    d = pd.DataFrame({"y": np.asarray(y).astype(int), "p": np.asarray(p), "period": np.asarray(period)})
    rows = []
    for k, g in d.groupby("period", sort=True):
        rows.append({"period": k, "orders": len(g), "base_rate_%": g.y.mean() * 100,
                     "pr_auc": average_precision_score(g.y, g.p) if g.y.any() else np.nan,
                     "roc_auc": roc_auc_score(g.y, g.p) if 0 < g.y.mean() < 1 else np.nan,
                     "lift": (average_precision_score(g.y, g.p) / g.y.mean()) if g.y.any() else np.nan})
    return pd.DataFrame(rows).set_index("period")


def bootstrap_diff(y, p_a, p_b, metric=average_precision_score, n: int = 1000,
                   seed: int = C.SEED) -> dict:
    """Paired bootstrap of metric(a) - metric(b): the same resampled rows score both models."""
    y = np.asarray(y).astype(int)
    p_a, p_b = np.asarray(p_a), np.asarray(p_b)
    rng = np.random.default_rng(seed)
    diffs = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, len(y), len(y))
        diffs[i] = metric(y[idx], p_a[idx]) - metric(y[idx], p_b[idx])
    return {"diff": metric(y, p_a) - metric(y, p_b), "ci_low": np.percentile(diffs, 2.5),
            "ci_high": np.percentile(diffs, 97.5), "share_a_better": (diffs > 0).mean()}


def bootstrap_ci(y, p, metric=average_precision_score, n: int = 1000, seed: int = C.SEED) -> tuple:
    y, p = np.asarray(y).astype(int), np.asarray(p)
    rng = np.random.default_rng(seed)
    vals = [metric(y[i], p[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(n))]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def precision_at_k(y, p, k: int) -> float:
    """Share of positives among the k highest-scored rows."""
    order = np.argsort(-np.asarray(p), kind="mergesort")[:k]
    return float(np.asarray(y)[order].mean())


def daily_top_k(y, p, day, k: int) -> dict:
    """Contact the k riskiest rows of every day: precision and share of all positives caught."""
    d = pd.DataFrame({"y": np.asarray(y).astype(int), "p": np.asarray(p), "day": np.asarray(day)})
    top = d.sort_values("p", ascending=False).groupby("day").head(k)
    return {"k_per_day": k, "contacted": len(top), "precision": top.y.mean(),
            "recall": top.y.sum() / max(d.y.sum(), 1), "base_rate": d.y.mean()}


def results_row(name: str, split: str, y, p, threshold: float) -> dict:
    return {"model": name, "split": split, "threshold": threshold, **scores(y, p, threshold)}
