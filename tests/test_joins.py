import pandas as pd
import pytest

from olist.joins import JoinError, safe_merge


@pytest.fixture
def orders():
    return pd.DataFrame({"order_id": ["a", "b", "c"], "x": [1, 2, 3]})


def test_valid_many_to_one_keeps_row_count(orders):
    dim = pd.DataFrame({"order_id": ["a", "b"], "y": [10, 20]})
    out = safe_merge(orders, dim, on="order_id", validate="1:1", name="t")
    assert len(out) == 3
    assert out.attrs["join_report"]["unmatched_left"] == 1
    assert "_merge_status" not in out.columns


def test_duplicate_right_keys_are_rejected(orders):
    items = pd.DataFrame({"order_id": ["a", "a", "b"], "price": [1, 2, 3]})
    with pytest.raises(JoinError):
        safe_merge(orders, items, on="order_id", validate="m:1")


def test_one_to_many_validation_is_refused(orders):
    items = pd.DataFrame({"order_id": ["a"], "price": [1]})
    with pytest.raises(JoinError, match="aggregate"):
        safe_merge(orders, items, on="order_id", validate="1:m")


def test_unmatched_keys_can_be_forbidden(orders):
    dim = pd.DataFrame({"order_id": ["a", "b"], "y": [10, 20]})
    with pytest.raises(JoinError):
        safe_merge(orders, dim, on="order_id", validate="1:1", allow_unmatched=False)


def test_left_on_right_on(orders):
    dim = pd.DataFrame({"id": ["a", "b", "c"], "y": [1, 2, 3]})
    out = safe_merge(orders, dim, left_on="order_id", right_on="id", validate="1:1")
    assert len(out) == 3 and out["y"].tolist() == [1, 2, 3]
