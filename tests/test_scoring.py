import pytest

from olist.scoring import MODEL_PATH, load_deployed, score


@pytest.mark.data
def test_deployed_model_scores_unseen_orders(order_dataset):
    if not MODEL_PATH.exists():
        pytest.skip("deployed model not built yet (run notebooks 06 and 08)")
    model, meta = load_deployed()
    df, _ = order_dataset
    open_orders = df[~df.order_status.isin(["delivered", "canceled", "unavailable"])].head(200)
    out = score(open_orders, model, meta)
    assert len(out) == len(open_orders)
    assert out.p_late.between(0, 1).all()
    assert set(out.action) <= {"EXPEDITE", "standard"}
    assert ((out.p_late > out.break_even_p) == (out.action == "EXPEDITE")).all()
