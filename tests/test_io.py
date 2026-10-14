import pandas as pd
import pytest

from olist import config as C
from olist.io import load, parse_dates


def test_day_first_parsing_is_explicit():
    s = pd.Series(["02-10-2017 10:56", "13-02-2018 21:18", None], name="ts")
    out = parse_dates(s, C.ORDER_TS_FORMAT)
    assert out.iloc[0] == pd.Timestamp("2017-10-02 10:56")  # 2 Oct, not 10 Feb
    assert out.iloc[1] == pd.Timestamp("2018-02-13 21:18")
    assert pd.isna(out.iloc[2])


def test_malformed_dates_fail_loudly():
    with pytest.raises(ValueError):
        parse_dates(pd.Series(["2017-10-02 10:56:00"], name="ts"), C.ORDER_TS_FORMAT)


def test_orders_loader_types_and_first_row():
    o = load("orders", nrows=50)
    for col in C.ORDER_TS_COLS:
        assert pd.api.types.is_datetime64_any_dtype(o[col]), col
    first = o.iloc[0]
    assert first["order_id"] == "e481f51cbdc54678b7cc49136f2d6af7"
    assert first["order_purchase_timestamp"] == pd.Timestamp("2017-10-02 10:56")
    assert (o["order_estimated_delivery_date"] == o["order_estimated_delivery_date"].dt.normalize()).all()


def test_zip_prefixes_keep_leading_zeros():
    c = load("customers")
    assert c["customer_zip_code_prefix"].str.len().eq(5).all()


def test_products_columns_renamed():
    p = load("products", nrows=5)
    assert "product_name_length" in p.columns and "product_name_lenght" not in p.columns
