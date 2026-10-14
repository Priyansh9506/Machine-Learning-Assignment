"""Project-wide constants: paths, file names, column subsets, dtypes and date formats.

Everything that controls *what* is read from disk lives here so that loaders stay
selective and reproducible.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "Dataset"
INTERIM_DIR = ROOT / "data" / "interim"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

SEED = 42

# The supplied copy stores orders/items timestamps as day-first "DD-MM-YYYY HH:MM"
# (seconds dropped). Reviews keep ISO dates. Always parse with an explicit format:
# pandas' inference silently mis-reads day-first strings such as "02-10-2017".
ORDER_TS_FORMAT = "%d-%m-%Y %H:%M"
REVIEW_TS_FORMAT = "ISO8601"

# Effective delivery-observation cutoff: weekly deliveries fall from ~1,600 to 28
# after this date (see notebooks/01_data_audit.ipynb, section 6).
T_OBS = pd.Timestamp("2018-09-03")

# Proposed analytical window for Task A training/evaluation rows (purchase date).
POPULATION_START = pd.Timestamp("2017-01-01")
POPULATION_END = pd.Timestamp("2018-08-01")  # exclusive

FILES = {
    "orders": "olist_orders_dataset.csv",
    "customers": "olist_customers_dataset.csv",
    "items": "olist_order_items_dataset.csv",
    "payments": "olist_order_payments_dataset.csv",
    "reviews": "olist_order_reviews_dataset.csv",
    "products": "olist_products_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "geolocation": "olist_geolocation_dataset.csv",
    "translation": "product_category_name_translation.csv",
}

ORDER_TS_COLS = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
]

# Default column subsets per table (None = all columns). Zip prefixes are read as
# strings so leading zeros survive.
USECOLS: dict[str, list[str] | None] = {
    "orders": ["order_id", "customer_id", "order_status", *ORDER_TS_COLS],
    "customers": ["customer_id", "customer_unique_id", "customer_zip_code_prefix",
                  "customer_city", "customer_state"],
    "items": ["order_id", "order_item_id", "product_id", "seller_id",
              "shipping_limit_date", "price", "freight_value"],
    "payments": ["order_id", "payment_sequential", "payment_type",
                 "payment_installments", "payment_value"],
    "reviews": ["review_id", "order_id", "review_score",
                "review_creation_date", "review_answer_timestamp"],
    "products": None,
    "sellers": None,
    "geolocation": ["geolocation_zip_code_prefix", "geolocation_lat",
                    "geolocation_lng", "geolocation_state"],
    "translation": None,
}

DTYPES: dict[str, dict[str, str]] = {
    "orders": {"order_status": "category"},
    "customers": {"customer_zip_code_prefix": "string", "customer_state": "category"},
    "items": {"order_item_id": "int8", "price": "float32", "freight_value": "float32"},
    "payments": {"payment_type": "category", "payment_sequential": "int8",
                 "payment_installments": "int8", "payment_value": "float32"},
    "reviews": {"review_score": "int8"},
    "products": {},
    "sellers": {"seller_zip_code_prefix": "string", "seller_state": "category"},
    "geolocation": {"geolocation_zip_code_prefix": "string", "geolocation_lat": "float32",
                    "geolocation_lng": "float32", "geolocation_state": "category"},
    "translation": {},
}

DATE_COLS = {
    "orders": (ORDER_TS_COLS, ORDER_TS_FORMAT),
    "items": (["shipping_limit_date"], ORDER_TS_FORMAT),
    "reviews": (["review_creation_date", "review_answer_timestamp"], REVIEW_TS_FORMAT),
}

# Source columns with typos, renamed on load.
RENAMES = {
    "products": {
        "product_name_lenght": "product_name_length",
        "product_description_lenght": "product_description_length",
    }
}

# The translation table misses two categories present in products.
EXTRA_TRANSLATIONS = {
    "pc_gamer": "pc_gamer",
    "portateis_cozinha_e_preparadores_de_alimentos": "portable_kitchen_and_food_preparers",
}

# Rough bounding box for Brazil, used to drop impossible geolocation points.
BRAZIL_BBOX = {"lat": (-34.0, 5.5), "lng": (-74.0, -34.0)}

PROCESSED_DIR = ROOT / "data" / "processed"
ORDER_DATASET_PATH = PROCESSED_DIR / "orders_model.parquet"

# Statuses of orders that were neither delivered nor cancelled (still "open" at extraction).
OPEN_STATUSES = ["shipped", "invoiced", "processing", "approved", "created"]

# As-of history features
SMOOTHING_M = 20          # pseudo-observations pulling a seller/route rate toward the prior
COLD_START_N = 5          # fewer resolved prior orders than this = cold start
PRIOR_FALLBACK = 0.05     # assumed prior late rate before ANY outcome is known (2016 only)

# Provisional chronological split on purchase date (end-exclusive); revisited after EDA.
SPLITS = {
    "train": (POPULATION_START, pd.Timestamp("2018-03-01")),
    "val": (pd.Timestamp("2018-03-01"), pd.Timestamp("2018-05-01")),
    "test": (pd.Timestamp("2018-05-01"), POPULATION_END),
}
