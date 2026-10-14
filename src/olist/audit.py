"""Data-audit helpers: profiles, key checks, orphan checks and timestamp anomalies."""
from __future__ import annotations

import pandas as pd


def profile(df: pd.DataFrame) -> pd.DataFrame:
    """Per-column dtype, missingness and cardinality."""
    return pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "n_null": df.isna().sum(),
        "pct_null": (df.isna().mean() * 100).round(2),
        "n_unique": df.nunique(),
    })


def table_summary(df: pd.DataFrame) -> dict:
    return {"rows": len(df), "cols": df.shape[1],
            "mem_mb": round(df.memory_usage(deep=True).sum() / 1e6, 2),
            "exact_dup_rows": int(df.duplicated().sum())}


def key_uniqueness(df: pd.DataFrame, keys: list) -> pd.DataFrame:
    """For each candidate key (column name or list of names): is it unique?"""
    rows = []
    for k in keys:
        cols = [k] if isinstance(k, str) else list(k)
        dups = int(df.duplicated(cols).sum())
        rows.append({"key": " + ".join(cols), "distinct": len(df) - dups,
                     "duplicate_rows": dups, "unique": dups == 0})
    return pd.DataFrame(rows).set_index("key")


def relationship(child: pd.DataFrame, parent: pd.DataFrame, child_key: str,
                 parent_key: str | None = None, name: str = "") -> dict:
    """Foreign-key coverage and observed cardinality between two tables."""
    parent_key = parent_key or child_key
    ck, pk = child[child_key], parent[parent_key]
    per_parent = ck.value_counts()
    return {
        "relationship": name or f"{child_key} -> {parent_key}",
        "child_rows": len(child),
        "child_keys_in_parent_pct": round(ck.isin(pk).mean() * 100, 3),
        "orphan_child_rows": int((~ck.isin(pk)).sum()),
        "parents_without_child": int((~pk.isin(ck)).sum()),
        "max_children_per_parent": int(per_parent.max()),
        "parent_key_unique": not pk.duplicated().any(),
        "cardinality": "1:1" if per_parent.max() == 1 else "1:m",
    }


def timestamp_anomalies(orders: pd.DataFrame) -> pd.DataFrame:
    """Count logically impossible orderings of the order lifecycle timestamps."""
    P, A, CR, DL, E = ("order_purchase_timestamp", "order_approved_at",
                       "order_delivered_carrier_date", "order_delivered_customer_date",
                       "order_estimated_delivery_date")
    checks = {
        "approved < purchase": orders[A] < orders[P],
        "carrier handoff < purchase": orders[CR] < orders[P],
        "carrier handoff < approval": orders[CR] < orders[A],
        "delivered < carrier handoff": orders[DL] < orders[CR],
        "delivered < purchase": orders[DL] < orders[P],
        "estimated < purchase": orders[E] < orders[P],
        "status delivered but no delivery date": (orders["order_status"] == "delivered") & orders[DL].isna(),
        "status not delivered but has delivery date": (orders["order_status"] != "delivered") & orders[DL].notna(),
    }
    return pd.DataFrame({
        "n_orders": {k: int(v.sum()) for k, v in checks.items()},
        "n_delivered_orders": {k: int((v & (orders["order_status"] == "delivered")).sum())
                               for k, v in checks.items()},
    })


def late_labels(orders: pd.DataFrame) -> pd.DataFrame:
    """Day-level and timestamp-level late labels (NaN where delivery is unobserved).

    The estimated date is always 00:00, so the timestamp rule effectively counts
    delivery *on* the promised day as late.
    """
    dl = orders["order_delivered_customer_date"]
    est = orders["order_estimated_delivery_date"]
    observed = dl.notna()
    return pd.DataFrame({
        "late_day": (dl.dt.normalize() > est).where(observed).astype("Float64"),
        "late_ts": (dl > est).where(observed).astype("Float64"),
    }, index=orders.index)
