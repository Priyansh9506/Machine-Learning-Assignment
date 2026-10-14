import numpy as np
import pandas as pd
import pytest

from olist.geo import build_zip_map, haversine_km, lookup_coords
from olist.io import load


def test_haversine_sao_paulo_rio():
    d = haversine_km(-23.55, -46.63, -22.91, -43.17)
    assert 350 < d < 370


def test_lookup_fallback_order():
    zm = pd.DataFrame({"lat": [-10.0, -12.0], "lng": [-40.0, -42.0], "n_points": [1, 1],
                       "state": ["BA", "BA"]}, index=pd.Index(["40000", "40010"], name="zip_prefix"))
    zips = pd.Series(["40000", "40099", "99999", "88888"])
    states = pd.Series(["BA", "BA", "BA", "XX"])
    out = lookup_coords(zips, states, zm)
    assert out["geo_source"].tolist() == ["zip5", "zip3", "state", "missing"]
    assert out.loc[0, "lat"] == -10.0
    assert out.loc[1, "lat"] == pytest.approx(-11.0)
    assert np.isnan(out.loc[3, "lat"])


@pytest.mark.data
def test_real_coverage_after_fallback():
    zm, _ = build_zip_map()
    c = load("customers")
    s = load("sellers")
    cc = lookup_coords(c["customer_zip_code_prefix"], c["customer_state"], zm)
    sc = lookup_coords(s["seller_zip_code_prefix"], s["seller_state"], zm)
    assert (cc["geo_source"] == "zip5").mean() >= 0.997
    assert (cc["geo_source"] != "missing").all() and (sc["geo_source"] != "missing").all()
    assert zm["lat"].between(-34, 5.5).all()
