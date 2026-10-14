"""Point-in-time ("as-of") historical features.

An order's outcome becomes *known* at
    known_at = min(delivered_ts, estimated_date + 1 day)
- on-time orders are known when they are delivered;
- late orders are known at the end of the promised day (not delivered by then);
- never-delivered, non-cancelled orders are known late at the end of the promised day.
Counting only orders "delivered before purchase" would under-count late orders, because
they resolve later.

A feature for an order purchased at t uses only events with known_at < t (strict).
Every count is computed with pd.merge_asof on cumulative sums: O(n log n), no loops.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


def asof_counts(queries: pd.DataFrame, events: pd.DataFrame, key: str | None, t_col: str,
                windows: tuple[int | None, ...] = (None,), event_time: str = "known_at",
                value: str = "late") -> pd.DataFrame:
    """For each query row, count events with the same key and event_time < t (strictly).

    Returns, aligned to queries.index, columns n_<w> and sum_<w> for each window, where
    w=None means all history ("all") and w=k means events in [t - k days, t).
    """
    if not queries.index.is_unique:
        raise ValueError("queries index must be unique")
    k = key or "_all"
    ev = events.dropna(subset=[event_time])
    ev = (ev.assign(_all=0) if key is None else ev)[[k, event_time, value]]
    ev = ev.sort_values(event_time, kind="mergesort").reset_index(drop=True)
    ev["_n"] = ev.groupby(k).cumcount() + 1
    ev["_s"] = ev.groupby(k)[value].cumsum()
    right = ev[[k, event_time, "_n", "_s"]]

    q = queries.assign(_all=0) if key is None else queries
    q = q[[k, t_col]].rename_axis("_qid").reset_index()

    def cum_before(t: pd.Series) -> pd.DataFrame:
        left = q.assign(_t=t.to_numpy()).sort_values("_t", kind="mergesort")
        m = pd.merge_asof(left, right, left_on="_t", right_on=event_time, by=k,
                          allow_exact_matches=False, direction="backward")
        return m.set_index("_qid")[["_n", "_s"]].fillna(0).reindex(q["_qid"])

    now = cum_before(q[t_col])
    out = {}
    for w in windows:
        name = "all" if w is None else f"{w}d"
        if w is None:
            n, s = now["_n"], now["_s"]
        else:
            past = cum_before(q[t_col] - pd.Timedelta(days=w))
            n, s = now["_n"] - past["_n"], now["_s"] - past["_s"]
        out[f"n_{name}"] = n.to_numpy()
        out[f"sum_{name}"] = s.to_numpy()
    return pd.DataFrame(out, index=queries.index)


def smoothed_rate(late: pd.Series, n: pd.Series, prior: pd.Series | float,
                  m: float = C.SMOOTHING_M) -> pd.Series:
    """Bayesian (beta-binomial) shrinkage of late/n toward the prior with m pseudo-counts."""
    return (late + m * prior) / (n + m)


def outcome_events(orders: pd.DataFrame) -> pd.DataFrame:
    """Order-level outcome events: order_id, known_at, late (from olist.dataset.add_labels)."""
    ev = orders.loc[orders["known_at"].notna(), ["order_id", "known_at", "event_late"]]
    return ev.rename(columns={"event_late": "late"})


def build_history_features(orders: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    """All as-of history features, one row per order_id in `pairs`.

    orders: order-level frame with order_id, order_purchase_timestamp, known_at, event_late,
            customer_state, customer_unique_id.
    pairs:  distinct (order_id, seller_id, seller_state) rows from order_items.
    """
    t = "order_purchase_timestamp"
    o = orders[["order_id", t, "known_at", "event_late", "customer_state", "customer_unique_id"]]
    ev_orders = outcome_events(o)

    # --- market-wide prior and recent market late rate --------------------------------
    oq = o.set_index("order_id")[[t, "customer_state", "customer_unique_id"]]
    g = asof_counts(oq, ev_orders, key=None, t_col=t, windows=(None, 30))
    prior = (g["sum_all"] / g["n_all"]).where(g["n_all"] > 0, C.PRIOR_FALLBACK)
    feats = pd.DataFrame(index=oq.index)
    feats["market_late_rate_30d"] = smoothed_rate(g["sum_30d"], g["n_30d"], prior)
    feats["market_resolved_30d"] = g["n_30d"]

    # --- destination state ------------------------------------------------------------
    ev_dest = ev_orders.merge(o[["order_id", "customer_state"]], on="order_id", validate="1:1")
    d = asof_counts(oq, ev_dest, key="customer_state", t_col=t, windows=(None, 90))
    feats["dest_state_late_rate"] = smoothed_rate(d["sum_all"], d["n_all"], prior)
    feats["dest_state_late_rate_90d"] = smoothed_rate(d["sum_90d"], d["n_90d"], prior)

    # --- customer: prior orders placed (purchase-time events, outcome not needed) -------
    ev_cust = o[["customer_unique_id", t]].rename(columns={t: "event_t"}).assign(one=1)
    cu = asof_counts(oq, ev_cust, key="customer_unique_id", t_col=t, event_time="event_t", value="one")
    feats["customer_prior_orders"] = cu["n_all"]

    # --- seller and route (computed per order-seller pair, then aggregated) -------------
    p = pairs.merge(o[["order_id", t, "customer_state"]], on="order_id", validate="m:1")
    p["route"] = p["seller_state"].astype(str) + ">" + p["customer_state"].astype(str)
    p = p.reset_index(drop=True)
    p_prior = p["order_id"].map(prior)

    ev_pairs = p[["order_id", "seller_id", "route"]].merge(ev_orders, on="order_id", validate="m:1")
    s = asof_counts(p, ev_pairs, key="seller_id", t_col=t, windows=(None, 90))
    r = asof_counts(p, ev_pairs, key="route", t_col=t, windows=(None,))
    act = asof_counts(p, p[["seller_id", t]].rename(columns={t: "event_t"}).assign(one=1),
                      key="seller_id", t_col=t, windows=(None, 7, 30), event_time="event_t", value="one")

    pf = pd.DataFrame({
        "order_id": p["order_id"],
        "seller_late_rate": smoothed_rate(s["sum_all"], s["n_all"], p_prior),
        "seller_late_rate_90d": smoothed_rate(s["sum_90d"], s["n_90d"], p_prior),
        "seller_resolved_n": s["n_all"],
        "route_late_rate": smoothed_rate(r["sum_all"], r["n_all"], p_prior),
        "route_resolved_n": r["n_all"],
        "seller_prior_orders": act["n_all"],
        "seller_orders_7d": act["n_7d"],
        "seller_orders_30d": act["n_30d"],
    })
    agg = pf.groupby("order_id").agg(
        seller_late_rate_max=("seller_late_rate", "max"),
        seller_late_rate_90d_max=("seller_late_rate_90d", "max"),
        seller_resolved_n_min=("seller_resolved_n", "min"),
        route_late_rate_max=("route_late_rate", "max"),
        route_resolved_n_min=("route_resolved_n", "min"),
        seller_prior_orders_min=("seller_prior_orders", "min"),
        seller_orders_7d_max=("seller_orders_7d", "max"),
        seller_orders_30d_max=("seller_orders_30d", "max"),
    )
    agg["seller_cold_start"] = (agg["seller_resolved_n_min"] < C.COLD_START_N).astype("int8")
    feats = feats.join(agg, how="inner")
    feats["history_prior"] = prior.reindex(feats.index)
    return feats.rename_axis("order_id").reset_index()
