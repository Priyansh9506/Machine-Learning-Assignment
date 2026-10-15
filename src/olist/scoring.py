"""Score orders with the deployed Task A model and turn probabilities into actions.

Used by predict.py. The deployed artefact (models/task_a_deploy.joblib) is a calibrated
wrapper around a fitted Pipeline; the meta file stores the feature list and cost model.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import config as C
from .decision import CostModel

MODEL_PATH = C.ROOT / "models" / "task_a_deploy.joblib"
META_PATH = C.ROOT / "models" / "task_a_deploy_meta.json"


def load_deployed(model_path: Path = MODEL_PATH, meta_path: Path = META_PATH):
    if not Path(model_path).exists():
        raise FileNotFoundError(f"{model_path} not found: run notebooks 06 and 08 (or `python run_all.py`) first")
    return joblib.load(model_path), json.loads(Path(meta_path).read_text())


def inner_pipeline(model):
    """The fitted Pipeline inside CalibratedClassifierCV(FrozenEstimator(pipeline))."""
    est = model.calibrated_classifiers_[0].estimator
    return getattr(est, "estimator", est)


def top_reasons(model, X: pd.DataFrame, k: int = 3) -> list[str]:
    """The k features pushing each order's risk up the most (tree SHAP or linear weight x value)."""
    pipe = inner_pipeline(model)
    pre, clf = pipe.named_steps["pre"], pipe.named_steps["clf"]
    Xt = pre.transform(X)
    names = np.asarray(Xt.columns)
    if hasattr(clf, "booster_"):
        contrib = clf.predict(Xt, pred_contrib=True)[:, :-1]
    elif hasattr(clf, "coef_"):
        contrib = Xt.to_numpy(dtype=float) * clf.coef_[0]
    else:
        return [""] * len(X)
    out = []
    for i, row in enumerate(contrib):
        idx = [j for j in np.argsort(-row)[:k] if row[j] > 0]
        parts = []
        for j in idx:
            raw = X.iloc[i][names[j]] if names[j] in X.columns else Xt.iloc[i, j]
            val = f"{raw:.3g}" if isinstance(raw, (float, np.floating)) else str(raw)
            parts.append(f"{names[j]}={val}")
        out.append("; ".join(parts))
    return out


def score(orders: pd.DataFrame, model, meta: dict, reasons: bool = True) -> pd.DataFrame:
    """Late probability, the per-order break-even threshold and the recommended action."""
    feats = meta["features"]
    cost = CostModel(**meta["cost_model"])
    p = model.predict_proba(orders[feats])[:, 1]
    thr = cost.threshold(orders)
    E, L = cost.expedite_cost(orders), cost.late_damage(orders)
    out = pd.DataFrame({
        "order_id": orders["order_id"].to_numpy(),
        "purchased": orders["order_purchase_timestamp"].to_numpy(),
        "promised_date": orders["order_estimated_delivery_date"].to_numpy(),
        "status": orders["order_status"].to_numpy() if "order_status" in orders else None,
        "route": (orders["main_seller_state"].astype(str) + " -> " + orders["customer_state"].astype(str)).to_numpy(),
        "p_late": p.round(4),
        "break_even_p": thr.round(4),
        "action": np.where(p > thr, "EXPEDITE", "standard"),
        "expected_value_of_expediting_brl": (p * cost.e * L - E).round(2),
    })
    if reasons:
        out["top_risk_factors"] = top_reasons(model, orders[feats].reset_index(drop=True))
    return out.sort_values("p_late", ascending=False).reset_index(drop=True)
