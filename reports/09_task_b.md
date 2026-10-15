# Task B report: bad-review risk at delivery

Evidence: [`notebooks/09_task_b_review_risk.ipynb`](../notebooks/09_task_b_review_risk.ipynb). Tables: `reports/results/09_*.csv`. Figure: `reports/figures/09_task_b.png`.

## Set-up

* **Prediction moment:** delivery confirmation, before the review request.
* **Target:** latest review score ≤ 2.
* **Population:** Task A orders with a review created **on or after the delivery day** (84,411 of 89,173 reviewed orders).
* **Inputs:** the 42 Task A features plus 8 delivery facts: actual delivery days, days late, late flag, delivered-vs-promised ratio, approval lag, dispatch days, transit days and delivery weekday.
* **Split by delivery date:** train before 22 Feb 2018 (49,305 orders), validation Mar–Apr (12,814), test May onward (20,723, 8.9% bad).

## Key finding: for late orders, the review request comes before the parcel

| | Orders | Request before delivery | Median request vs promise | Bad-review rate |
|---|---:|---:|---:|---:|
| Late orders | 5,993 | **75.8%** | 2 days after | 63.7% |
| On-time orders | 83,180 | 0.3% | 12 days before | 9.4% |

Late customers asked before delivery leave 1–2★ **75.7%** of the time, against 26.2% when asked after delivery. For three quarters of late orders, a post-delivery intervention is too late. **Task A, at checkout, is the only lever against lateness.** As a result, Task B's population contains few late orders (1.35% in training), so its dissatisfaction is mostly about other things.

## Results on test

| Model | Validation PR-AUC | Test PR-AUC | Test ROC-AUC | Precision of 20 calls/day | Bad reviews reached with 20 calls/day |
|---|---:|---:|---:|---:|---:|
| majority (≈ random calls) | 0.112 | 0.089 | 0.500 | 0.105 | 11.2% |
| rule: delivered late | 0.129 | 0.093 | 0.511 | 0.119 | 12.7% |
| logreg: days_late only | 0.155 | 0.098 | 0.500 | 0.112 | 12.0% |
| logistic regression | 0.294 | 0.195 | 0.656 | 0.229 | 24.4% |
| **LightGBM** (Optuna, 40 trials) | 0.293 | **0.196** | 0.652 | **0.230** | **24.5%** |

**Where the signal comes from:**

| Features | Test PR-AUC | Precision of 20 calls/day |
|---|---:|---:|
| checkout features only | 0.181 | 0.221 |
| delivery facts only | 0.126 | 0.138 |
| both | 0.196 | 0.230 |

* **Top SHAP feature: `n_items`.** Multi-item orders are a major source of unhappiness, plausibly because they arrive split or partially. It is followed by actual delivery days, delivered-vs-promised and dispatch delay.
* **Stable over time:** lift is 1.9–2.5× in every test month, because the bad-review rate barely drifts (8.6–9.4%).
* **Use:** a daily call list for support, 2–3× better than random at every list size. With 5 calls a day, 31% of the customers contacted are future 1–2★ reviewers.
