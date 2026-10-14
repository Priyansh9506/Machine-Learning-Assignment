"""Temporal-leakage tests for the as-of history features."""
import numpy as np
import pandas as pd
import pytest

from olist.history import asof_counts, build_history_features


def _toy():
    t = pd.Timestamp("2018-01-01")
    ev = pd.DataFrame({"k": ["a", "a", "a", "b", "a"],
                       "known_at": [t, t + pd.Timedelta(days=1), t + pd.Timedelta(days=1), t, t + pd.Timedelta(days=40)],
                       "late": [1.0, 0.0, 1.0, 1.0, 1.0]})
    q = pd.DataFrame({"k": ["a", "a", "a", "b", "c"],
                      "t": [t, t + pd.Timedelta(days=1), t + pd.Timedelta(days=2), t + pd.Timedelta(days=5),
                            t + pd.Timedelta(days=5)]}, index=[10, 11, 12, 13, 14])
    return ev, q


def test_strictly_before_and_ties():
    ev, q = _toy()
    out = asof_counts(q, ev, key="k", t_col="t", windows=(None,))
    # event at exactly t is NOT visible; tied events at day+1 are visible together
    assert out.loc[10, "n_all"] == 0
    assert out.loc[11, "n_all"] == 1 and out.loc[11, "sum_all"] == 1
    assert out.loc[12, "n_all"] == 3 and out.loc[12, "sum_all"] == 2
    assert out.loc[13, "n_all"] == 1 and out.loc[14, "n_all"] == 0


def test_window_counts():
    ev, q = _toy()
    q2 = pd.DataFrame({"k": ["a"], "t": [pd.Timestamp("2018-02-15")]})
    out = asof_counts(q2, ev, key="k", t_col="t", windows=(None, 30))
    assert out.loc[0, "n_all"] == 4 and out.loc[0, "n_30d"] == 1  # only the day-40 event is within 30d


def _brute(q_row, ev, key, t_col, window=None):
    m = (ev[key] == q_row[key]) & (ev["known_at"] < q_row[t_col])
    if window is not None:
        m &= ev["known_at"] >= q_row[t_col] - pd.Timedelta(days=window)
    return m.sum(), ev.loc[m, "late"].sum()


def test_matches_brute_force_on_random_data():
    rng = np.random.default_rng(0)
    base = pd.Timestamp("2017-01-01")
    n = 3000
    ev = pd.DataFrame({"k": rng.choice(list("abcdef"), n),
                       "known_at": base + pd.to_timedelta(rng.integers(0, 400 * 24, n), unit="h"),
                       "late": rng.integers(0, 2, n).astype(float)})
    q = pd.DataFrame({"k": rng.choice(list("abcdefg"), 400),
                      "t": base + pd.to_timedelta(rng.integers(0, 400 * 24, 400), unit="h")})
    out = asof_counts(q, ev, key="k", t_col="t", windows=(None, 30))
    for i, row in q.iterrows():
        assert (out.loc[i, "n_all"], out.loc[i, "sum_all"]) == _brute(row, ev, "k", "t")
        assert (out.loc[i, "n_30d"], out.loc[i, "sum_30d"]) == _brute(row, ev, "k", "t", 30)


@pytest.mark.data
def test_real_features_ignore_future_outcomes(order_dataset_inputs):
    """Scramble every outcome that becomes known on/after a cutoff: features of orders
    purchased before the cutoff must not change at all."""
    orders, pairs = order_dataset_inputs
    cutoff = pd.Timestamp("2018-01-01")
    base = build_history_features(orders, pairs).set_index("order_id")

    rng = np.random.default_rng(1)
    future = orders["known_at"] >= cutoff
    perturbed = orders.copy()
    perturbed.loc[future, "event_late"] = rng.integers(0, 2, future.sum()).astype(float)
    # also move future known_at times around (still >= cutoff)
    perturbed.loc[future, "known_at"] = cutoff + pd.to_timedelta(rng.integers(0, 2000, future.sum()), unit="h")
    alt = build_history_features(perturbed, pairs).set_index("order_id")

    before = orders.set_index("order_id")["order_purchase_timestamp"] < cutoff
    ids = base.index[before.reindex(base.index).fillna(False).to_numpy()]
    pd.testing.assert_frame_equal(base.loc[ids], alt.loc[ids])
    # sanity: the perturbation does change later orders
    assert not base.loc[~base.index.isin(ids)].equals(alt.loc[~alt.index.isin(ids)])


@pytest.mark.data
def test_real_seller_features_match_brute_force(order_dataset_inputs):
    from olist import config as C
    from olist.history import smoothed_rate
    orders, pairs = order_dataset_inputs
    feats = build_history_features(orders, pairs).set_index("order_id")
    ev = pairs.merge(orders[["order_id", "known_at", "event_late"]], on="order_id").dropna(subset=["known_at"])
    t = orders.set_index("order_id")["order_purchase_timestamp"]
    rng = np.random.default_rng(2)
    single = pairs.groupby("order_id").filter(lambda g: len(g) == 1)
    sample = single.sample(150, random_state=3)
    for _, row in sample.iterrows():
        ts = t[row.order_id]
        past = ev[(ev.seller_id == row.seller_id) & (ev.known_at < ts)]
        assert feats.loc[row.order_id, "seller_resolved_n_min"] == len(past)
        prior = feats.loc[row.order_id, "history_prior"]
        expected = smoothed_rate(past.event_late.sum(), len(past), prior, C.SMOOTHING_M)
        assert feats.loc[row.order_id, "seller_late_rate_max"] == pytest.approx(expected)
