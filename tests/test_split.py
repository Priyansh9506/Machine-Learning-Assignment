import pandas as pd
import pytest

from olist.split import assign_split, matured_by


def test_maturity_purge_ignores_outcome():
    cut = pd.Timestamp("2018-03-01")
    df = pd.DataFrame({
        "order_estimated_delivery_date": pd.to_datetime(["2018-02-27", "2018-02-28", "2018-03-05", "2018-03-05"]),
        # fast on-time vs pending order with the same promise: both must be treated alike
        "known_at": pd.to_datetime(["2018-02-20", "2018-03-01", "2018-02-10", "2018-03-06"]),
    })
    assert matured_by(df, cut).tolist() == [True, True, False, False]
    assert (df.loc[matured_by(df, cut), "known_at"] <= cut).all()


def test_assign_split_boundaries():
    splits = {"a": (pd.Timestamp("2018-01-01"), pd.Timestamp("2018-02-01")),
              "b": (pd.Timestamp("2018-02-01"), pd.Timestamp("2018-03-01"))}
    s = assign_split(pd.Series(pd.to_datetime(["2017-12-31", "2018-01-01", "2018-02-01", "2018-03-01"])), splits)
    assert s.tolist()[1:3] == ["a", "b"] and s.isna().tolist() == [True, False, False, True]
