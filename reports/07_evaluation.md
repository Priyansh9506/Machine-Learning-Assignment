# Stage 11 report: evaluation, explainability and error analysis

Evidence: [`notebooks/07_evaluation_and_explainability.ipynb`](../notebooks/07_evaluation_and_explainability.ipynb) (executed; about 1 min). Tables: `reports/results/07_*.csv`. Figures: `reports/figures/07_*.png`.

Test period: purchases May–Jul 2018, **18,944 orders, 722 late (3.81%)**. Thresholds were chosen on validation and applied unchanged.

## Results on the test period

| Model | Validation PR-AUC | **Test PR-AUC** [95% CI] | Test ROC-AUC | Brier | Precision | Recall | F1 | Flagged |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LightGBM | 0.217 | 0.088 [0.077, 0.103] | 0.700 | 0.036 | 0.086 | 0.460 | 0.144 | 20.5% |
| Random forest | 0.201 | 0.084 [0.073, 0.097] | 0.691 | 0.037 | 0.086 | 0.382 | 0.140 | 17.0% |
| **Logistic regression** (selected) | **0.221** | 0.083 [0.073, 0.095] | 0.690 | 0.037 | 0.082 | 0.366 | 0.134 | 16.9% |
| Decision tree | 0.161 | 0.051 [0.046, 0.056] | 0.613 | 0.037 | 0.055 | 0.402 | 0.096 | 27.9% |
| **Baseline: core logreg (5 features)** | 0.138 | **0.110 [0.094, 0.132]** | **0.707** | 0.036 | 0.087 | 0.446 | 0.146 | 19.5% |
| Baseline: business rule | 0.125 | 0.042 | 0.542 | 0.403 | 0.046 | 0.482 | 0.084 | 40.2% |
| Baseline: majority | 0.118 | 0.038 | 0.500 | 0.037 | 0 | 0 | 0 | 0% |

**Headline.** The selected model ranks late orders at **2.2× the base rate** and catches 37% of them while flagging 17% of orders. It clearly beats the majority and business-rule baselines. **But the simple 5-feature baseline beats every tuned model on test**, and the paired bootstrap confirms the gap is real: logistic regression is 0.028 lower, 95% CI [−0.047, −0.011]. LightGBM and logistic regression are statistically tied (+0.005, CI [−0.004, +0.015]).

## Disclosure: the test set was looked at twice

A first evaluation used 47 features, including five "clock" features, and a LightGBM with the early-stopping bug:

| Model | First pass (47 features) | Final (42 features) |
|---|---:|---:|
| LightGBM | 0.047 | 0.088 |
| Random forest | 0.057 | 0.084 |
| Logistic regression | 0.078 | 0.083 |
| Decision tree | 0.049 | 0.051 |

The clock features were found and removed using label-free checks (notebook 04, section 6). Iteration then **stopped**, so that the test set would not become a second validation set. The final numbers are somewhat optimistic compared with a single blind evaluation.

## Why does the simple baseline win? (post-hoc diagnosis)

| Model | Validation PR-AUC | Test PR-AUC | Test precision in top 5% |
|---|---:|---:|---:|
| core logreg, 5 features (with `purchase_month`) | 0.138 | 0.110 | 0.146 |
| core logreg, 4 features (**no month**) | 0.196 | 0.109 | 0.136 |
| tuned logistic regression, 42 features | 0.221 | 0.083 | 0.118 |
| tuned LightGBM, 42 features | 0.217 | 0.088 | 0.124 |

The edge does not come from the month dummies. It comes from **parsimony**: distance, promise length, seller late rate and freight ratio. Under the regime change from 11.8% late (validation) to 3.8% (test), the 42-feature models' extra geography (27 state dummies and regional rates) stopped transferring, while the physical delivery features stayed stable. **Lesson:** select models across several time periods, and prefer compact, physically meaningful features under drift.

## Calibration

The models keep predicting about 5–7% whatever the period. They under-predict validation (11.8% actual) and over-predict test (3.8% actual; logistic regression predicts 6.9%). Test Brier scores (0.036–0.037) barely beat a constant prediction (0.0367). **Ranking is useful; the probability level does not follow drift**, which is addressed in Stage 12.

## Per test month (logistic regression)

| Month | Late % | PR-AUC | Lift |
|---|---:|---:|---:|
| 2018-05 | 6.56 | 0.146 | 2.2× |
| 2018-06 | 1.17 | 0.042 | 3.6× |
| 2018-07 | 3.41 | 0.055 | 1.6× |

## Explainability

* **Logistic regression weights** (`07_global_importance.png`): the elastic net keeps only **24 of 96** weights non-zero. The largest are the Black Friday window (+0.70), destination RJ (+0.23), freight (+0.14), promised business days (−0.12), longitude (+0.12), distance (+0.10) and the destination-state late rate (+0.09).
* **Permutation importance on validation** (logistic regression): promised business days, promised days, distance, longitude, item count and freight. The promise and the trip dominate.
* **Decision tree rules** (`07_tree_rules.png`): the first split is the Black Friday window, then the (target-encoded) destination state, then promised business days.
* **SHAP for LightGBM** (`07_shap_beeswarm.png`): a high-risk destination state, short promises and a poor seller history push risk up; long promises pull it down.
* **Local cases** (`07_shap_local.png`):
  - *Correct alert:* SP → AL, 1,934 km, 29-day promise, p = 0.30. Destination +0.87 log-odds, seller history +0.24.
  - *Missed late order:* SP → PE, 2,102 km, **39-day promise**, p = 0.05. The long promise (−0.25, −0.22) outweighed state and seller.
  - *False alarm:* SP → AL, 21-day promise, p = 0.32, delivered on time. Destination is a strong but blunt signal.

## Error analysis

* **Slices** (`07_error_slices.csv`): lift is about 2.2–2.7× in most slices. Northeast orders are flagged wholesale (recall 0.91, precision 0.11). **The South is weakest** (lift 1.5, recall 0.17). Cold-start sellers are handled as well as established ones (lift 2.4 vs 2.3).
* **The 20 most confident errors are all missed late orders** (p ≈ 0.025–0.03). They are short South/Southeast trips with single items, ordinary promises and reliable sellers. Their lateness comes from events after checkout that no checkout-time feature can see. This is the irreducible part of the task.
