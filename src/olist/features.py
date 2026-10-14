"""Feature registry and leakage audit for Task A (prediction moment = order purchase).

Every model input must be listed in FEATURES. FORBIDDEN lists columns in the dataset
that must never be inputs; tests assert the two sets are disjoint.
"""
from __future__ import annotations

import pandas as pd

# name, group, kind (num/cat/bin), source, when the value is known
_SPEC = [
    # promise and calendar
    ("promised_days", "promise", "num", "orders.estimated - purchase", "checkout (estimate shown to customer)"),
    ("promised_business_days", "promise", "num", "orders + holiday calendar", "checkout"),
    ("holidays_in_promise", "promise", "num", "orders + holiday calendar", "checkout"),
    ("promise_dow", "promise", "cat", "orders.estimated", "checkout"),
    ("purchase_hour", "calendar", "num", "orders.purchase", "checkout"),
    ("purchase_dow", "calendar", "cat", "orders.purchase", "checkout"),
    ("purchase_month", "calendar", "cat", "orders.purchase", "checkout"),
    ("purchase_dom", "calendar", "num", "orders.purchase", "checkout"),
    ("purchase_is_weekend", "calendar", "bin", "orders.purchase", "checkout"),
    ("black_friday_period", "calendar", "bin", "orders.purchase + calendar", "checkout"),
    ("holiday_within_7d", "calendar", "bin", "orders.purchase + calendar", "checkout"),
    # basket
    ("n_items", "basket", "num", "items (aggregated)", "checkout"),
    ("n_products", "basket", "num", "items (aggregated)", "checkout"),
    ("n_sellers", "basket", "num", "items (aggregated)", "checkout"),
    ("n_categories", "basket", "num", "items+products (aggregated)", "checkout"),
    ("total_price", "basket", "num", "items (aggregated)", "checkout"),
    ("max_item_price", "basket", "num", "items (aggregated)", "checkout"),
    ("total_freight", "basket", "num", "items (aggregated)", "checkout"),
    ("freight_ratio", "basket", "num", "items (aggregated)", "checkout"),
    ("any_zero_freight", "basket", "bin", "items (aggregated)", "checkout"),
    # product
    ("total_weight_g", "product", "num", "products via items", "catalogue (static)"),
    ("max_weight_g", "product", "num", "products via items", "catalogue (static)"),
    ("total_volume_cm3", "product", "num", "products via items", "catalogue (static)"),
    ("max_volume_cm3", "product", "num", "products via items", "catalogue (static)"),
    ("mean_photos", "product", "num", "products via items", "catalogue (static)"),
    ("mean_description_length", "product", "num", "products via items", "catalogue (static)"),
    ("mean_name_length", "product", "num", "products via items", "catalogue (static)"),
    ("main_category", "product", "cat", "products via items", "catalogue (static)"),
    ("any_category_missing", "product", "bin", "products via items", "catalogue (static)"),
    # geography
    ("customer_state", "geography", "cat", "customers", "checkout"),
    ("main_seller_state", "geography", "cat", "sellers via items", "checkout"),
    ("n_seller_states", "geography", "num", "sellers via items", "checkout"),
    ("all_same_state", "geography", "bin", "sellers + customers", "checkout"),
    ("distance_km_max", "geography", "num", "zip map (geolocation)", "checkout"),
    ("distance_km_min", "geography", "num", "zip map (geolocation)", "checkout"),
    ("customer_lat", "geography", "num", "zip map (geolocation)", "checkout"),
    ("customer_lng", "geography", "num", "zip map (geolocation)", "checkout"),
    ("customer_geo_imputed", "geography", "bin", "zip map (geolocation)", "checkout"),
    ("seller_geo_imputed", "geography", "bin", "zip map (geolocation)", "checkout"),
    # payment (ablation-gated)
    ("main_payment_type", "payment", "cat", "payments (aggregated)", "checkout"),
    ("n_payment_types", "payment", "num", "payments (aggregated)", "checkout"),
    ("max_installments", "payment", "num", "payments (aggregated)", "checkout"),
    ("uses_voucher", "payment", "bin", "payments (aggregated)", "checkout"),
    # as-of history
    ("seller_late_rate_max", "history", "num", "as-of outcomes, known_at < purchase", "as-of purchase"),
    ("seller_late_rate_90d_max", "history", "num", "as-of outcomes, known_at in [t-90d, t)", "as-of purchase"),
    ("seller_resolved_n_min", "history", "num", "as-of outcomes, known_at < purchase", "as-of purchase"),
    ("seller_cold_start", "history", "bin", "as-of outcomes, known_at < purchase", "as-of purchase"),
    ("seller_prior_orders_min", "history", "num", "prior purchases < purchase", "as-of purchase"),
    ("seller_orders_7d_max", "history", "num", "prior purchases in [t-7d, t)", "as-of purchase"),
    ("seller_orders_30d_max", "history", "num", "prior purchases in [t-30d, t)", "as-of purchase"),
    ("route_late_rate_max", "history", "num", "as-of outcomes by state pair", "as-of purchase"),
    ("route_resolved_n_min", "history", "num", "as-of outcomes by state pair", "as-of purchase"),
    ("dest_state_late_rate", "history", "num", "as-of outcomes by customer state", "as-of purchase"),
    ("dest_state_late_rate_90d", "history", "num", "as-of outcomes by customer state", "as-of purchase"),
    ("market_late_rate_30d", "history", "num", "as-of outcomes, all orders", "as-of purchase"),
    ("market_resolved_30d", "history", "num", "as-of outcomes, all orders", "as-of purchase"),
    ("customer_prior_orders", "history", "num", "prior purchases by customer_unique_id", "as-of purchase"),
]
FEATURES = pd.DataFrame(_SPEC, columns=["feature", "group", "kind", "source", "known_when"]).set_index("feature")

# Present in the dataset but never allowed as Task A inputs.
FORBIDDEN = {
    "order_approved_at": "after payment approval",
    "order_delivered_carrier_date": "dispatch (post-checkout)",
    "order_delivered_customer_date": "delivery = the outcome",
    "order_status": "final status (outcome-dependent)",
    "known_at": "derived from the outcome",
    "event_late": "outcome",
    "late_day": "target", "late_ts": "target", "promise_failed": "target",
    "outcome_class": "outcome",
    "eligible_primary": "outcome-dependent filter", "eligible_sensitivity": "outcome-dependent filter",
    "dq_carrier_before_purchase": "post-checkout data-quality flag",
    "dq_carrier_before_approval": "post-checkout data-quality flag",
    "dq_delivered_before_carrier": "post-delivery data-quality flag",
    "order_estimated_delivery_date": "raw date; used only through promise features",
    "order_purchase_timestamp": "raw timestamp; used only through calendar features and splits",
    "history_prior": "global as-of trend; diagnostic only",
}
EXCLUDED_SOURCES = {
    "items.shipping_limit_date": "set after approval (~approval + 6 days)",
    "payments.payment_value": "redundant with price + freight",
    "reviews.*": "post-delivery",
}


def feature_names(groups: list[str] | None = None, kinds: list[str] | None = None) -> list[str]:
    f = FEATURES
    if groups is not None:
        f = f[f["group"].isin(groups)]
    if kinds is not None:
        f = f[f["kind"].isin(kinds)]
    return f.index.tolist()


def leakage_audit() -> pd.DataFrame:
    """One table of every candidate column: when it is known and whether it is a Task A input."""
    allowed = FEATURES.assign(task_a_input="yes")[["group", "source", "known_when", "task_a_input"]]
    forb = pd.DataFrame({"group": "excluded", "source": "orders / derived", "known_when": FORBIDDEN.values(),
                         "task_a_input": "no"}, index=list(FORBIDDEN))
    excl = pd.DataFrame({"group": "excluded", "source": "not loaded", "known_when": EXCLUDED_SOURCES.values(),
                         "task_a_input": "no"}, index=list(EXCLUDED_SOURCES))
    return pd.concat([allowed, forb, excl]).rename_axis("column")
