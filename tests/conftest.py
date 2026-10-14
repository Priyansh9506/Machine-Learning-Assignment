import pytest


@pytest.fixture(scope="session")
def order_dataset_inputs():
    """Labelled orders (with customer columns) and order-seller pairs from the real data."""
    from olist.dataset import add_labels, load_sources
    src = load_sources()
    o = add_labels(src["orders"]).merge(src["customers"], on="customer_id", validate="1:1")
    o["customer_state"] = o["customer_state"].astype(str)
    pairs = (src["items"][["order_id", "seller_id"]].drop_duplicates()
             .merge(src["sellers"][["seller_id", "seller_state"]], on="seller_id", validate="m:1"))
    pairs["seller_state"] = pairs["seller_state"].astype(str)
    return o, pairs


@pytest.fixture(scope="session")
def order_dataset():
    from olist.dataset import build_order_dataset
    return build_order_dataset()
