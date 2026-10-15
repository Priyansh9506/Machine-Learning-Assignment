import numpy as np
import pandas as pd
import pytest

from olist.baselines import BusinessRule, MajorityBaseline
from olist.decision import CostModel, policy_table, realised_saving, sensitivity
from olist.evaluate import bootstrap_diff, daily_top_k, precision_at_k, scores


@pytest.fixture
def orders():
    return pd.DataFrame({"total_price": [100.0, 200.0, 50.0, 400.0], "total_freight": [20.0, 10.0, 30.0, 40.0]})


def test_threshold_is_break_even(orders):
    c = CostModel(a=0.5, handling=2.0, b=0.5, e=0.5)
    t = c.threshold(orders)
    E, L = c.expedite_cost(orders), c.late_damage(orders)
    assert np.allclose(t * c.e * L - E, 0)                       # expected value is exactly zero at t
    assert t[0] == pytest.approx((2 + 10) / (0.5 * 0.5 * 120))


def test_realised_saving_counts_only_acted_orders(orders):
    c = CostModel()
    y = np.array([1, 0, 0, 1])
    act = np.array([True, True, False, False])
    expected = (1 * c.e * c.late_damage(orders)[0] - c.expedite_cost(orders)[0]) - c.expedite_cost(orders)[1]
    assert realised_saving(y, act, orders, c) == pytest.approx(expected)
    assert realised_saving(y, np.zeros(4, bool), orders, c) == 0


def test_policy_table_and_sensitivity_shapes(orders):
    y, p = np.array([1, 0, 0, 1]), np.array([0.9, 0.1, 0.2, 0.6])
    pt = policy_table(y, p, orders, CostModel(), global_threshold=0.5)
    assert pt.loc["do nothing", "saving_BRL"] == 0
    assert len(sensitivity(y, p, orders, CostModel())) == 8


def test_scores_and_ranking_helpers():
    y = np.array([0, 0, 1, 1, 0, 1])
    p = np.array([0.1, 0.2, 0.8, 0.7, 0.3, 0.9])
    s = scores(y, p, 0.5)
    assert s["pr_auc"] == pytest.approx(1.0) and s["recall"] == 1 and s["precision"] == 1
    assert precision_at_k(y, p, 3) == 1.0
    day = np.array(["d1", "d1", "d1", "d2", "d2", "d2"])
    assert daily_top_k(y, p, day, 1)["precision"] == 1.0
    b = bootstrap_diff(y, p, 1 - p, n=50)
    assert b["diff"] > 0 and b["ci_low"] <= b["diff"] <= b["ci_high"]


def test_baselines_behave():
    X = pd.DataFrame({"seller_late_rate_max": np.r_[np.full(50, 0.02), np.full(50, 0.3)],
                      "distance_km_max": np.r_[np.full(50, 100.0), np.full(50, 2000.0)]})
    y = np.r_[np.zeros(50), np.ones(50)]
    assert np.allclose(MajorityBaseline().fit(X, y).predict_proba(X)[:, 1], 0.5)
    rule = BusinessRule().fit(X, y)
    assert rule.train_f1_ == 1.0 and (rule.predict(X) == y).all()
