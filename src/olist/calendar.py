"""Brazilian national holidays 2016-2018 and Black Friday dates (hard-coded; no extra dependency)."""
from __future__ import annotations

import numpy as np
import pandas as pd

_FIXED = ["01-01", "04-21", "05-01", "09-07", "10-12", "11-02", "11-15", "12-25"]
_MOVABLE = {  # Carnival Mon/Tue, Good Friday, Corpus Christi
    2016: ["2016-02-08", "2016-02-09", "2016-03-25", "2016-05-26"],
    2017: ["2017-02-27", "2017-02-28", "2017-04-14", "2017-06-15"],
    2018: ["2018-02-12", "2018-02-13", "2018-03-30", "2018-05-31"],
    2019: ["2019-03-04", "2019-03-05", "2019-04-19", "2019-06-20"],
}
BLACK_FRIDAY = pd.to_datetime(["2016-11-25", "2017-11-24", "2018-11-23"])

HOLIDAYS = pd.DatetimeIndex(sorted(
    [pd.Timestamp(f"{y}-{md}") for y in _MOVABLE for md in _FIXED]
    + [pd.Timestamp(d) for v in _MOVABLE.values() for d in v]))


def holidays_between(start: pd.Series, end: pd.Series) -> np.ndarray:
    """Number of holidays in the half-open day interval [start, end)."""
    h = HOLIDAYS.values.astype("datetime64[D]")
    s = start.values.astype("datetime64[D]")
    e = end.values.astype("datetime64[D]")
    return np.searchsorted(h, e, side="left") - np.searchsorted(h, s, side="left")


def business_days_between(start: pd.Series, end: pd.Series) -> np.ndarray:
    """Mon-Fri non-holiday days in [start, end)."""
    return np.busday_count(start.values.astype("datetime64[D]"), end.values.astype("datetime64[D]"),
                           holidays=HOLIDAYS.values.astype("datetime64[D]"))


def black_friday_period(ts: pd.Series, before: int = 3, after: int = 7) -> pd.Series:
    """True when the purchase falls within [BF - before, BF + after] days of that year's Black Friday."""
    d = ts.dt.normalize()
    out = pd.Series(False, index=ts.index)
    for bf in BLACK_FRIDAY:
        out |= d.between(bf - pd.Timedelta(days=before), bf + pd.Timedelta(days=after))
    return out
