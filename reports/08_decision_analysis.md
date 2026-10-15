# Stage 12 report: cost-sensitive decisions, drift and deployment

Evidence: [`notebooks/08_decision_analysis.ipynb`](../notebooks/08_decision_analysis.ipynb). Tables: `reports/results/08_*.csv`. Code: `src/olist/decision.py`, `src/olist/scoring.py`, `predict.py`.

## The decision rule

| Quantity | Definition | Default (assumption) |
|---|---|---|
| Expedite cost `E_i` | 2 BRL + a × freight | a = 0.5 → median 11.2 BRL |
| Late damage `L_i` | b × (price + freight) | b = 0.5 → median 54.8 BRL |
| Effectiveness `e` | share of late orders that expediting saves | 0.5 |

**Act when `p_i > E_i / (e · L_i)`**, which is `t* = C / (C + B)` with `C = E_i` and `B = e·L_i − E_i`. The median break-even probability is **0.46**.

## Calibration

| Probabilities | Test Brier | Mean predicted | Actual |
|---|---:|---:|---:|
| raw logistic regression | 0.037 | 6.9% | 3.8% |
| sigmoid, fitted on validation | 0.044 | 11.4% | 3.8% |
| isotonic, fitted on validation | 0.044 | 11.2% | 3.8% |

Calibrating on an 11.8%-late period imports that regime into a 3.8%-late one.

## Policies on the test period

| Policy | Orders acted on | Late orders caught | BRL per 1,000 orders |
|---|---:|---:|---:|
| do nothing | 0% | 0% | **0** |
| expedite every order | 100% | 100% | −12,497 |
| business rule | 40.2% | 48.2% | −5,673 |
| model, one threshold chosen on validation (0.49) | 0% | 0% | 0 |
| **model, per-order cost rule** | 5.5% | 10.9% | **−293** |

The same per-order rule on validation (11.8% late) **saves +470 BRL per 1,000 orders**. The model loses about 40× less than expediting everything, and it picks sensible orders: the ones it expedites have a median value of 464 BRL against 81, and are twice as likely to be late (7.6% vs 3.6%). But in a low-lateness month, expediting at these costs is not worth it.

## Sensitivity

Each assumption moved ±50% (`08_tornado.png`): the model policy loses between 100 and 529 BRL per 1,000 test orders. It always beats expediting everything, but never beats doing nothing in the test period.

**Break-even map** (`08_break_even_map.png`):
* On validation the model pays off almost everywhere (+470 at the defaults, +2,305 at b = 1).
* On test it pays off only where expediting works at least 75% of the time **and** a late order costs about 1.75–2× its value (+260 to +2,305).

## Drift: static vs monthly recalibration vs monthly retraining

| Month | Late % | Static PR-AUC | Retrained PR-AUC | Static BRL/1,000 | Recalibrated | Retrained |
|---|---:|---:|---:|---:|---:|---:|
| 2018-05 | 6.6 | 0.146 | 0.154 | −91 | −164 | −370 |
| 2018-06 | 1.2 | 0.042 | 0.071 | −595 | −137 | −126 |
| 2018-07 | 3.4 | 0.055 | 0.063 | −215 | −105 | −27 |

The static model over-predicts every month (10.4–12.0% predicted against 1.2–6.6% actual). Recalibration fixes the level. Retraining also improves the ranking: mean PR-AUC 0.096 against 0.081.

## Recommendations

1. **Retrain monthly and recalibrate on the most recent orders.** The deployed artefact follows this recipe.
2. **Expedite only when lateness is running high**, and only orders where a late delivery is genuinely costly.
3. **Use the model for near-zero-cost actions** (an honest ETA, a proactive message), where the break-even probability is low and the model helps in every period.
4. **Measure `e` with an A/B test** before spending money on expediting. The data is observational, so the benefit is assumed, not proven.

## Deployment

`models/task_a_deploy.joblib`: logistic regression refitted on 83,696 orders purchased up to 30 Jun 2018, calibrated (sigmoid) on July 2018, which was held out (6,099 orders, 3.41% late). `predict.py` rebuilds all features with the training code and outputs `p_late`, the per-order break-even probability, the action and the top risk factors.

Example run on the 1,721 orders still in transit at the snapshot: mean risk 3.1%; **3 orders** recommended for expediting. Output in `reports/predictions.csv`.
