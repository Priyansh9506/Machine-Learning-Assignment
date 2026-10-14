"""Integrity of the order-level dataset and of the feature registry."""
import pandas as pd
import pytest

from olist import config as C
from olist.dataset import add_labels, aggregate_items
from olist.features import FEATURES, FORBIDDEN, leakage_audit


def test_registry_and_forbidden_are_disjoint():
    assert not set(FEATURES.index) & set(FORBIDDEN)
    assert FEATURES.index.is_unique
    assert set(FEATURES["kind"]) <= {"num", "cat", "bin"}
    audit = leakage_audit()
    assert set(audit.loc[audit.task_a_input == "yes"].index) == set(FEATURES.index)


def test_labels_and_known_at_on_toy_orders():
    ts = lambda s: pd.Timestamp(s)
    o = pd.DataFrame({
        "order_id": ["on_time", "on_day", "late", "open", "cancel"],
        "order_status": ["delivered", "delivered", "delivered", "shipped", "canceled"],
        "order_purchase_timestamp": [ts("2018-01-01")] * 5,
        "order_approved_at": [ts("2018-01-01 01:00")] * 5,
        "order_delivered_carrier_date": [ts("2018-01-02")] * 5,
        "order_delivered_customer_date": [ts("2018-01-05 10:00"), ts("2018-01-10 15:00"), ts("2018-01-12 09:00"), pd.NaT, pd.NaT],
        "order_estimated_delivery_date": [ts("2018-01-10")] * 5,
    })
    lab = add_labels(o).set_index("order_id")
    assert lab.loc["on_time", "late_day"] == 0 and lab.loc["on_day", "late_day"] == 0
    assert lab.loc["on_day", "late_ts"] == 1          # on-day counts late only under the timestamp rule
    assert lab.loc["late", "late_day"] == 1
    assert lab.loc["on_time", "known_at"] == ts("2018-01-05 10:00")
    assert lab.loc["late", "known_at"] == ts("2018-01-11")          # end of promised day, not delivery
    assert lab.loc["open", "promise_failed"] == 1 and pd.isna(lab.loc["open", "late_day"])
    assert pd.isna(lab.loc["cancel", "promise_failed"]) and pd.isna(lab.loc["cancel", "known_at"])


def test_aggregate_items_one_row_per_order():
    it = pd.DataFrame({"order_id": ["a", "a", "b"], "order_item_id": [1, 2, 1], "product_id": ["p", "q", "p"],
                       "seller_id": ["s", "t", "s"], "price": [10.0, 30.0, 5.0], "freight_value": [2.0, 0.0, 1.0],
                       "product_category_name": ["x", None, "x"], "product_name_length": [1, 1, 1],
                       "product_description_length": [1, 1, 1], "product_photos_qty": [1, 1, 1],
                       "product_weight_g": [100.0, None, None], "volume_cm3": [10.0, 20.0, None],
                       "seller_state": ["SP", "RJ", "SP"], "seller_geo_source": ["zip5"] * 3,
                       "distance_km": [10.0, 400.0, 5.0], "same_state": [True, False, True]})
    agg = aggregate_items(it).set_index("order_id")
    assert len(agg) == 2
    assert agg.loc["a", "n_sellers"] == 2 and agg.loc["a", "distance_km_max"] == 400
    assert agg.loc["a", "main_category"] == "unknown"      # 30.0 item has no category
    assert agg.loc["a", "total_weight_g"] == 100 and pd.isna(agg.loc["b", "total_weight_g"])
    assert agg.loc["a", "freight_ratio"] == pytest.approx(2 / 40)


@pytest.mark.data
def test_real_dataset_integrity(order_dataset):
    df, joins = order_dataset
    assert df["order_id"].is_unique
    assert all(j["rows_after"] == j["rows_before"] for j in joins)
    assert set(FEATURES.index) <= set(df.columns)
    pop = df[df["eligible_primary"]]
    assert len(pop) == 89_795 and int(pop["late_day"].sum()) == 6_138
    assert pop["late_day"].notna().all()
    assert (pop["order_purchase_timestamp"] >= C.POPULATION_START).all()
    assert (pop["order_estimated_delivery_date"] <= C.T_OBS).all()
    # outcome becomes known strictly after the prediction moment
    k = df.dropna(subset=["known_at"])
    assert (k["known_at"] > k["order_purchase_timestamp"]).all()
