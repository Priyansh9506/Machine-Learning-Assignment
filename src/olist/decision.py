"""Cost-sensitive decision layer for Task A.

For each order the marketplace can act (expedite shipping) or not.
    E_i = handling + a * freight_i               cost of expediting, paid whenever we act
    L_i = b * (price_i + freight_i)              damage of a late order (refund, support, lost repeat business)
    e   = P(expediting turns a late order into an on-time one)
Acting on order i is worth  y_i * e * L_i - E_i  (realised), or  p_i * e * L_i - E_i  in expectation.
So act exactly when p_i > t_i = E_i / (e * L_i), which is the textbook t* = C / (C + B) with
C = E_i (wasted cost on an on-time order) and B = e * L_i - E_i (net benefit on a late one).

All monetary parameters are ASSUMPTIONS, which is why `sensitivity` is part of the analysis.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CostModel:
    a: float = 0.5          # expedite cost as a fraction of the order's freight
    handling: float = 2.0   # fixed BRL per expedited order
    b: float = 0.5          # late-order damage as a fraction of order value (price + freight)
    e: float = 0.5          # effectiveness: share of late orders that expediting saves

    def expedite_cost(self, df: pd.DataFrame) -> np.ndarray:
        return self.handling + self.a * df["total_freight"].to_numpy(dtype=float)

    def late_damage(self, df: pd.DataFrame) -> np.ndarray:
        return self.b * (df["total_price"].to_numpy(dtype=float) + df["total_freight"].to_numpy(dtype=float))

    def threshold(self, df: pd.DataFrame) -> np.ndarray:
        """Per-order break-even probability t_i = E_i / (e * L_i)."""
        return self.expedite_cost(df) / (self.e * self.late_damage(df))

    def as_dict(self) -> dict:
        return asdict(self)


def realised_saving(y, act, df: pd.DataFrame, cost: CostModel) -> float:
    """Money saved versus doing nothing, using the actual outcomes y."""
    y = np.asarray(y, dtype=float)
    act = np.asarray(act, dtype=bool)
    value = y * cost.e * cost.late_damage(df) - cost.expedite_cost(df)
    return float(value[act].sum())


def per_1000(total: float, n: int) -> float:
    return total / n * 1000


def policy_table(y, p, df: pd.DataFrame, cost: CostModel, global_threshold: float | None = None,
                 rule_mask=None) -> pd.DataFrame:
    """Compare do-nothing, act-on-all, an optional business rule and model policies."""
    n = len(df)
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    policies = {"do nothing": np.zeros(n, bool), "act on every order": np.ones(n, bool)}
    if rule_mask is not None:
        policies["business rule"] = np.asarray(rule_mask, bool)
    if global_threshold is not None:
        policies[f"model, one threshold ({global_threshold:.2f})"] = p >= global_threshold
    policies["model, per-order cost rule"] = p > cost.threshold(df)
    rows = []
    for name, act in policies.items():
        s = realised_saving(y, act, df, cost)
        rows.append({"policy": name, "acted_%": act.mean() * 100,
                     "late_caught_%": (act & (y == 1)).sum() / max(y.sum(), 1) * 100,
                     "saving_BRL": s, "saving_per_1000_orders": per_1000(s, n)})
    return pd.DataFrame(rows).set_index("policy")


def profit_curve(y, p, df: pd.DataFrame, cost: CostModel, grid=None) -> pd.DataFrame:
    """Saving per 1,000 orders when acting on every order with p >= threshold."""
    grid = np.linspace(0, 1, 101) if grid is None else grid
    p = np.asarray(p, dtype=float)
    return pd.DataFrame({"threshold": grid,
                         "saving_per_1000": [per_1000(realised_saving(y, p >= t, df, cost), len(df)) for t in grid],
                         "acted_%": [(p >= t).mean() * 100 for t in grid]})


def sensitivity(y, p, df: pd.DataFrame, base: CostModel, swing: float = 0.5) -> pd.DataFrame:
    """Vary each cost parameter by +/- swing; the cost rule is re-derived for every scenario."""
    rows = []
    for field in ["a", "handling", "b", "e"]:
        for sign in (-1, 1):
            v = getattr(base, field) * (1 + sign * swing)
            if field == "e":
                v = min(v, 1.0)
            c = replace(base, **{field: v})
            act = np.asarray(p) > c.threshold(df)
            all_act = np.ones(len(df), bool)
            rows.append({"parameter": field, "change": f"{sign * swing:+.0%}", "value": v,
                         "model_saving_per_1000": per_1000(realised_saving(y, act, df, c), len(df)),
                         "act_all_saving_per_1000": per_1000(realised_saving(y, all_act, df, c), len(df)),
                         "acted_%": act.mean() * 100})
    out = pd.DataFrame(rows)
    out["model_beats_both"] = (out["model_saving_per_1000"] > 0) & \
                              (out["model_saving_per_1000"] > out["act_all_saving_per_1000"])
    return out
