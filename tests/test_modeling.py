import numpy as np
import pandas as pd
import pytest

from olist.features import FEATURES, FORBIDDEN, model_features
from olist.modeling import Winsorizer, linear_preprocessor, tree_preprocessor, walk_forward_folds


def _toy(n=400, seed=0):
    """A frame with every registered feature, plausible types and a binary target."""
    rng = np.random.default_rng(seed)
    d = {}
    for f, kind in FEATURES["kind"].items():
        if kind == "num":
            d[f] = rng.gamma(2.0, 10.0, n)
        elif kind == "bin":
            d[f] = rng.integers(0, 2, n)
        else:
            d[f] = rng.choice(["a", "b", "c"], n) if f in ("main_category", "customer_state", "main_seller_state",
                                                            "main_payment_type") else rng.integers(0, 7, n)
    df = pd.DataFrame(d)
    df.loc[:9, "total_weight_g"] = np.nan
    return df, pd.Series(rng.integers(0, 2, n))


def test_model_features_are_registered_and_never_forbidden():
    f = model_features()
    assert set(f) <= set(FEATURES.index)
    assert not set(f) & set(FORBIDDEN)


def test_winsorizer_uses_training_quantiles():
    train = pd.DataFrame({"x": np.arange(101, dtype=float)})
    w = Winsorizer(0.1, 0.9).fit(train)
    out = w.transform(pd.DataFrame({"x": [-1000.0, 50.0, 1000.0]}))
    assert out[0, 0] == pytest.approx(10) and out[1, 0] == 50 and out[2, 0] == pytest.approx(90)


def test_linear_preprocessor_is_complete_and_fitted_on_train_only():
    X, y = _toy()
    pre = linear_preprocessor(list(FEATURES.index)).fit(X.iloc[:300], y.iloc[:300])
    shifted = X.iloc[300:].copy()
    shifted["promised_days"] += 1000                       # a shift in new data must not be re-learned
    out = pre.transform(shifted)
    assert out.isna().sum().sum() == 0
    # clipped at the training 99th percentile and scaled with training statistics, not re-centred
    assert out["promised_days"].nunique() == 1 and out["promised_days"].iloc[0] > 1.5


def test_tree_preprocessor_keeps_one_column_per_feature():
    X, y = _toy()
    out = tree_preprocessor(list(FEATURES.index)).fit(X, y).transform(X)
    assert sorted(out.columns) == sorted(FEATURES.index)
    assert out["total_weight_g"].isna().sum() == 10       # trees receive NaN untouched


def test_walk_forward_folds_never_train_on_unknown_labels():
    days = pd.date_range("2017-06-01", "2018-02-28", freq="D")
    df = pd.DataFrame({"order_purchase_timestamp": np.repeat(days, 3)})
    df["order_estimated_delivery_date"] = df.order_purchase_timestamp.dt.normalize() + pd.to_timedelta(
        np.tile([5, 20, 40], len(days)), unit="D")
    for (tr, va), m in zip(walk_forward_folds(df), ["2017-11", "2017-12", "2018-01", "2018-02"]):
        start = pd.Timestamp(m)
        assert (df.order_purchase_timestamp.iloc[va].dt.to_period("M").astype(str) == m).all()
        assert (df.order_estimated_delivery_date.iloc[tr] + pd.Timedelta(days=1) <= start).all()
        assert not set(tr) & set(va)
