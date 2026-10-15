"""Build the order-level modelling dataset (one row per order that has items).

Pipeline (every join validated by safe_merge):
    items -m:1- products, items -m:1- sellers(+coords), items -m:1- order/customer(+coords)
    -> aggregate items to order grain
    orders -1:1- customers(+coords) -1:1- items_agg -1:1- payments_agg -1:1- history
Labels, population flags and the as-of `known_at` time are attached; no post-checkout
field is turned into a feature (see olist.features.FEATURES).
"""
from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd

from . import calendar as cal
from . import config as C
from .geo import build_zip_map, haversine_km, lookup_coords
from .history import build_history_features
from .io import load
from .joins import safe_merge

log = logging.getLogger(__name__)
P, A, CR, DL, E = C.ORDER_TS_COLS
ONE_DAY = pd.Timedelta(days=1)


def load_sources(data_dir=None) -> dict[str, pd.DataFrame]:
    """Only the tables and columns the primary pipeline needs (reviews/translation excluded)."""
    return {
        "orders": load("orders", data_dir=data_dir),
        "customers": load("customers", usecols=["customer_id", "customer_unique_id",
                                                "customer_zip_code_prefix", "customer_state"], data_dir=data_dir),
        "items": load("items", usecols=["order_id", "order_item_id", "product_id", "seller_id",
                                        "price", "freight_value"], data_dir=data_dir),
        "products": load("products", data_dir=data_dir),
        "sellers": load("sellers", usecols=["seller_id", "seller_zip_code_prefix", "seller_state"],
                        data_dir=data_dir),
        "payments": load("payments", data_dir=data_dir),
    }


def add_labels(orders: pd.DataFrame) -> pd.DataFrame:
    """Targets, outcome-known time and population flags (all retrospective, never features)."""
    o = orders.copy()
    snapshot = o[[P, A, CR, DL]].max().max()
    delivered = o["order_status"].eq("delivered") & o[DL].notna()
    open_failed = o["order_status"].isin(C.OPEN_STATUSES) & (o[E] + ONE_DAY <= snapshot)

    late_day = (o[DL].dt.normalize() > o[E]).astype(float)
    late_ts = (o[DL] > o[E]).astype(float)
    o["late_day"] = late_day.where(delivered)
    o["late_ts"] = late_ts.where(delivered)
    o["promise_failed"] = np.where(delivered, late_day, np.where(open_failed, 1.0, np.nan))
    o["outcome_class"] = np.select([delivered, open_failed], ["delivered", "open_failed"], "other")

    known_delivered = pd.concat([o[DL], o[E] + ONE_DAY], axis=1).min(axis=1)
    o["known_at"] = known_delivered.where(delivered, (o[E] + ONE_DAY).where(open_failed))
    o["event_late"] = o["promise_failed"]  # outcome used by history features

    window = (o[P] >= C.POPULATION_START) & (o[P] < C.POPULATION_END) & (o[E] <= C.T_OBS)
    o["eligible_primary"] = delivered & window
    o["eligible_sensitivity"] = (delivered | open_failed) & window
    # data-quality flags on post-checkout fields (kept for transparency, never features)
    o["dq_carrier_before_purchase"] = o[CR] < o[P]
    o["dq_carrier_before_approval"] = o[CR] < o[A]
    o["dq_delivered_before_carrier"] = o[DL] < o[CR]
    return o


def order_calendar_features(o: pd.DataFrame) -> pd.DataFrame:
    """Calendar and promise features, all known at checkout."""
    p, e = o[P], o[E]
    day = p.dt.normalize()
    f = pd.DataFrame(index=o.index)
    f["purchase_hour"] = p.dt.hour.astype("int8")
    f["purchase_dow"] = p.dt.dayofweek.astype("int8")
    f["purchase_month"] = p.dt.month.astype("int8")
    f["purchase_dom"] = p.dt.day.astype("int8")
    f["purchase_is_weekend"] = (f["purchase_dow"] >= 5).astype("int8")
    f["promised_days"] = (e - day).dt.days.astype("int16")
    f["promised_business_days"] = cal.business_days_between(day, e).astype("int16")
    f["holidays_in_promise"] = cal.holidays_between(day, e + ONE_DAY).astype("int8")
    f["promise_dow"] = e.dt.dayofweek.astype("int8")
    f["black_friday_period"] = cal.black_friday_period(p).astype("int8")
    f["holiday_within_7d"] = (cal.holidays_between(day, day + pd.Timedelta(days=8)) > 0).astype("int8")
    return f


def item_level(src: dict, zip_map: pd.DataFrame, cust_orders: pd.DataFrame) -> pd.DataFrame:
    """Item rows enriched with product attributes, seller coords and customer coords."""
    pr = src["products"].copy()
    pr["volume_cm3"] = pr["product_length_cm"] * pr["product_height_cm"] * pr["product_width_cm"]
    s = src["sellers"].copy()
    s["seller_state"] = s["seller_state"].astype(str)
    sc = lookup_coords(s["seller_zip_code_prefix"], s["seller_state"], zip_map)
    s[["seller_lat", "seller_lng", "seller_geo_source"]] = sc[["lat", "lng", "geo_source"]].to_numpy()

    it = safe_merge(src["items"], pr[["product_id", "product_category_name", "product_name_length",
                                      "product_description_length", "product_photos_qty",
                                      "product_weight_g", "volume_cm3"]],
                    on="product_id", validate="m:1", name="items-products")
    it = safe_merge(it, s[["seller_id", "seller_state", "seller_lat", "seller_lng", "seller_geo_source"]],
                    on="seller_id", validate="m:1", name="items-sellers")
    it = safe_merge(it, cust_orders[["order_id", "customer_state", "customer_lat", "customer_lng"]],
                    on="order_id", validate="m:1", name="items-orders", allow_unmatched=False)
    it["distance_km"] = haversine_km(it["seller_lat"].astype(float), it["seller_lng"].astype(float),
                                     it["customer_lat"].astype(float), it["customer_lng"].astype(float))
    it["same_state"] = it["seller_state"] == it["customer_state"]
    return it


def aggregate_items(it: pd.DataFrame) -> pd.DataFrame:
    """Collapse item rows to one row per order."""
    it = it.assign(_cat_missing=it["product_category_name"].isna(),
                   _zero_freight=it["freight_value"] == 0,
                   _seller_geo_imputed=it["seller_geo_source"] != "zip5")
    g = it.groupby("order_id")
    agg = g.agg(
        n_items=("order_item_id", "size"),
        n_products=("product_id", "nunique"),
        n_sellers=("seller_id", "nunique"),
        n_categories=("product_category_name", "nunique"),
        n_seller_states=("seller_state", "nunique"),
        total_price=("price", "sum"),
        max_item_price=("price", "max"),
        total_freight=("freight_value", "sum"),
        total_weight_g=("product_weight_g", "sum"),
        _n_weight=("product_weight_g", "count"),
        max_weight_g=("product_weight_g", "max"),
        total_volume_cm3=("volume_cm3", "sum"),
        _n_volume=("volume_cm3", "count"),
        max_volume_cm3=("volume_cm3", "max"),
        mean_photos=("product_photos_qty", "mean"),
        mean_description_length=("product_description_length", "mean"),
        mean_name_length=("product_name_length", "mean"),
        distance_km_max=("distance_km", "max"),
        distance_km_min=("distance_km", "min"),
        all_same_state=("same_state", "all"),
        any_category_missing=("_cat_missing", "any"),
        any_zero_freight=("_zero_freight", "any"),
        seller_geo_imputed=("_seller_geo_imputed", "any"),
    )
    # an order whose items all lack weight/volume gets NaN, not 0
    agg["total_weight_g"] = agg["total_weight_g"].where(agg.pop("_n_weight") > 0)
    agg["total_volume_cm3"] = agg["total_volume_cm3"].where(agg.pop("_n_volume") > 0)
    agg["freight_ratio"] = agg["total_freight"] / agg["total_price"]
    # attributes of the "main" item line (highest price; ties -> first line)
    main = it.sort_values(["order_id", "price", "order_item_id"], ascending=[True, False, True]) \
             .drop_duplicates("order_id").set_index("order_id")
    agg["main_category"] = main["product_category_name"].fillna("unknown")
    agg["main_seller_state"] = main["seller_state"]
    for c in ["all_same_state", "any_category_missing", "any_zero_freight", "seller_geo_imputed"]:
        agg[c] = agg[c].astype("int8")
    return agg.reset_index()


def aggregate_payments(pay: pd.DataFrame) -> pd.DataFrame:
    """Checkout payment choices, one row per order."""
    pay = pay.sort_values(["order_id", "payment_value"], ascending=[True, False])
    agg = pay.groupby("order_id").agg(
        main_payment_type=("payment_type", "first"),
        n_payment_types=("payment_type", "nunique"),
        n_payment_records=("payment_sequential", "size"),
        max_installments=("payment_installments", "max"),
    )
    agg["uses_voucher"] = pay.assign(v=pay["payment_type"] == "voucher").groupby("order_id")["v"].any().astype("int8")
    agg["main_payment_type"] = agg["main_payment_type"].astype(str)
    return agg.reset_index()


def build_order_dataset(src: dict | None = None, zip_map: pd.DataFrame | None = None
                        ) -> tuple[pd.DataFrame, list[dict]]:
    """Return (order-level dataset, join/row-count log)."""
    t0 = time.perf_counter()
    src = src or load_sources()
    zip_map = zip_map if zip_map is not None else build_zip_map()[0]
    joins: list[dict] = []

    o = add_labels(src["orders"])
    o["order_status"] = o["order_status"].astype(str)
    cu = src["customers"].copy()
    cu["customer_state"] = cu["customer_state"].astype(str)
    cc = lookup_coords(cu["customer_zip_code_prefix"], cu["customer_state"], zip_map)
    cu["customer_lat"], cu["customer_lng"] = cc["lat"].to_numpy(), cc["lng"].to_numpy()
    cu["customer_geo_imputed"] = (cc["geo_source"] != "zip5").astype("int8").to_numpy()
    o = safe_merge(o, cu.drop(columns="customer_zip_code_prefix"), on="customer_id", validate="1:1",
                   name="orders-customers", allow_unmatched=False)
    joins.append(o.attrs["join_report"])

    it = item_level(src, zip_map, o)
    items_agg = aggregate_items(it)
    o = safe_merge(o, items_agg, on="order_id", validate="1:1", name="orders-items_agg")
    joins.append(o.attrs["join_report"])
    o = safe_merge(o, aggregate_payments(src["payments"]), on="order_id", validate="1:1",
                   name="orders-payments_agg")
    joins.append(o.attrs["join_report"])

    o = pd.concat([o, order_calendar_features(o)], axis=1)

    pairs = it[["order_id", "seller_id", "seller_state"]].drop_duplicates()
    hist = build_history_features(o, pairs)
    o = safe_merge(o, hist, on="order_id", validate="1:1", name="orders-history")
    joins.append(o.attrs["join_report"])

    o["has_items"] = o["n_items"].notna()
    o = o[o["has_items"]].drop(columns="has_items").reset_index(drop=True)
    o.attrs = {}
    log.info("order dataset %s built in %.1fs", o.shape, time.perf_counter() - t0)
    return o, joins
