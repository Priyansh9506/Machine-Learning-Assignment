"""Chronological split on purchase date, with label-maturity purging."""
from __future__ import annotations

import pandas as pd

from . import config as C

ONE_DAY = pd.Timedelta(days=1)


def assign_split(purchase: pd.Series, splits: dict = C.SPLITS) -> pd.Series:
    out = pd.Series(pd.NA, index=purchase.index, dtype="string")
    for name, (start, end) in splits.items():
        out[(purchase >= start) & (purchase < end)] = name
    return out


def matured_by(df: pd.DataFrame, cutoff: pd.Timestamp) -> pd.Series:
    """True where the outcome is guaranteed to be known at `cutoff` *whatever it is*:
    the promised day has fully passed (estimated + 1 day <= cutoff).

    Purging on this checkout-known quantity removes recent long-promise orders without
    looking at the outcome. Purging on `known_at < cutoff` instead would keep recent
    orders only if they were delivered early, i.e. select on the label (on-time bias).
    """
    return df["order_estimated_delivery_date"] + ONE_DAY <= cutoff


def label_known_by(df: pd.DataFrame, cutoff: pd.Timestamp) -> pd.Series:
    """Outcome actually known before `cutoff` (outcome-dependent; diagnostics only)."""
    return df["known_at"] < cutoff


def split_summary(df: pd.DataFrame, target: str = "late_day", splits: dict = C.SPLITS) -> pd.DataFrame:
    """Counts, class ratios and maturity-purge counts per split."""
    rows = []
    s = assign_split(df["order_purchase_timestamp"], splits)
    for name, (start, end) in splits.items():
        part = df[s == name]
        mat = matured_by(part, end)
        rows.append({"split": name, "purchase_from": start.date(), "purchase_to_excl": end.date(),
                     "orders": len(part), "late": int(part[target].sum()),
                     "late_rate_%": round(part[target].mean() * 100, 2),
                     "matured_by_period_end": int(mat.sum()),
                     "late_rate_matured_%": round(part.loc[mat, target].mean() * 100, 2)})
    return pd.DataFrame(rows).set_index("split")
