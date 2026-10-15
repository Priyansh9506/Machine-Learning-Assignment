# Stage 8 report: baselines, a deliberately leaky model and decision boundaries

Evidence: [`notebooks/05_baselines_and_decision_boundaries.ipynb`](../notebooks/05_baselines_and_decision_boundaries.ipynb) (executed; about 35 s). Tables: `reports/results/05_*.csv`. Code: `src/olist/baselines.py`.

## Baselines (fitted on the matured training set)

| Baseline | CV PR-AUC (4 months) | Validation PR-AUC | Validation recall | Validation precision | Validation accuracy | Flagged |
|---|---:|---:|---:|---:|---:|---:|
| Majority class ("always on time") | 0.099 ± 0.035 | 0.118 | 0.000 | – | **88.2%** | 0% |
| Business rule (seller rate > 9.5% **or** distance > 1,342 km) | 0.107 ± 0.038 | 0.125 | 0.416 | 0.133 | 61.1% | 36.9% |
| Core logistic regression (5 features) | 0.142 ± 0.060 | 0.138 | 0.190 | 0.130 | 75.3% | 17.3% |

The majority baseline shows why accuracy is the wrong metric: 88% accurate and useless. The business rule barely beats the base rate on PR-AUC. The core logistic regression is the bar every Stage 9 model has to clear.

## What leakage would have done

Default LightGBM on validation:

| Inputs | PR-AUC | ROC-AUC |
|---|---:|---:|
| Audited features only (honest) | **0.188** | 0.644 |
| + seller late rate computed on the full data | 0.275 | 0.749 |
| + carrier-handoff days (post-checkout) | 0.282 | 0.705 |
| + both | 0.352 | 0.776 |

Two tempting columns inflate PR-AUC by **87%**, and all of it is fake.

## Decision boundaries and hyperplanes

| Figure | What it shows |
|---|---|
| `05_decision_boundaries_2d.png` | Four models trained on log-distance and promised days only. Logistic regression draws **one straight line** (a hyperplane in 2-D). The depth-4 tree draws **axis-aligned rectangles** (around 300 km and 1,300 km, and 11, 22 and 33 promised days). The forest and LightGBM draw **stepped, bendy frontiers** with small islands. |
| `05_hyperplane_3d.png` | Logistic regression on three standardised features. The boundary is a **plane**: `+0.42·dist − 0.38·days + 0.27·state − 2.91 = logit(0.07)` |
| `05_hyperplane_full_space.png` | The full model (96 columns after one-hot encoding of the 42 features). Left: signed distance of each validation order to the hyperplane. At p = 0.14, 23.2% of late orders vs 6.6% of on-time orders fall beyond the boundary. Right: the largest components of the normal vector `w`: the Black Friday window and destination-state dummies. |
