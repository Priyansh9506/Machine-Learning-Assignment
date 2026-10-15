# Stages 5–7 report: preprocessing, class imbalance and feature selection

Evidence: [`notebooks/04_preprocessing_and_imbalance.ipynb`](../notebooks/04_preprocessing_and_imbalance.ipynb). Tables: `reports/results/04_*.csv`. Code: `src/olist/modeling.py`.

## Walk-forward validation inside the training period

Each fold validates on one month and trains only on earlier orders whose promised day had already passed (the Stage 3 maturity rule).

| Validation month | Training rows | Last training purchase | Validation rows | Late % |
|---|---:|---|---:|---:|
| 2017-10 *(threshold-moving only)* | 23,141 | 2017-09-20 | 4,478 | 4.18 |
| 2017-11 | 27,153 | 2017-10-19 | 7,288 | 12.40 |
| 2017-12 | 31,684 | 2017-11-17 | 5,513 | 7.46 |
| 2018-01 | 38,762 | 2017-12-12 | 7,069 | 5.70 |
| 2018-02 | 44,227 | 2018-01-16 | 6,555 | 14.13 |

## Preprocessing (all fitted on training rows inside a `Pipeline`)

| Feature type | Linear models | Tree models |
|---|---|---|
| Skewed numerics (25) | log1p → median impute + missing flag → winsorise 1/99% → standardise | raw, NaN kept |
| Other numerics | median impute + flag → winsorise → standardise | raw |
| Binary flags | passthrough | passthrough |
| States, payment, weekdays | one-hot (levels with < 30 rows pooled) | target encoding (states), integer codes, ordinal (payment) |
| `main_category` | cross-fitted target encoding | cross-fitted target encoding (later dropped) |

The notebook shows that the imputer holds the *training* medians and that standardised validation columns are not mean 0 / sd 1, which proves nothing was refitted on validation. The linear design matrix has 129 columns (before feature selection) with no NaN. The tree matrix keeps one column per feature.

## Class imbalance: five strategies (mean of 4 monthly folds)

| Learner | Strategy | PR-AUC | Brier | F1 | Flagged |
|---|---|---:|---:|---:|---:|
| Logistic regression | none | **0.167** | **0.092** | 0.001 (t = 0.5) | 0.04% |
| | class weights | 0.159 | 0.277 | 0.219 | 48.5% |
| | undersample 1:1 | 0.156 | 0.200 | 0.212 | 29.4% |
| | SMOTE-NC 1:1 | 0.167 | 0.423 | 0.124 | 62.9% |
| | threshold moving | 0.167 | 0.092 | 0.132 | 35.0% |
| LightGBM | none | **0.161** | **0.092** | 0.004 (t = 0.5) | 0.04% |
| | class weights | 0.141 | 0.107 | 0.084 | 3.2% |
| | undersample 1:1 | 0.157 | 0.217 | 0.228 | 32.1% |
| | SMOTE-NC 1:1 | 0.128 | 0.136 | 0.055 | 9.8% |
| | threshold moving | 0.161 | 0.092 | 0.184 | 29.2% |

**Decision.** No strategy ranks better than a plain fit, and every resampling or weighting strategy damages the probabilities. The models are therefore trained on the natural class ratio. Imbalance is handled by PR-AUC as the selection metric, a threshold chosen on validation (Stage 11) and a cost rule on calibrated probabilities (Stage 12). Threshold moving with last month's threshold is unstable under drift (Jan 5.7% → Feb 14.1%), which is a warning for Stage 12.

## Feature selection

**Drop-one-group ablation** (Δ walk-forward PR-AUC when the group is removed; negative = the group helps):

| Removed group | Δ LightGBM | Δ logistic regression |
|---|---:|---:|
| geography | −0.016 | +0.004 |
| calendar (other) | −0.013 | +0.001 |
| promise | −0.012 | −0.010 |
| purchase_month | −0.004 | −0.020 |
| history: market | −0.002 | +0.002 |
| payment | −0.001 | −0.001 |
| redundant near-duplicates | +0.002 | +0.001 |
| history: seller | +0.003 | −0.000 |
| **product** | **+0.005** | **+0.001** |

The lasso (C = 0.02) zeroes 27 of 57 features, including weight and volume. LightGBM permutation importance on the Feb 2018 fold puts `main_category`, weights and volumes at about 0.

**The `purchase_month` paradox.** Every fold is a single month, so within a fold the month is constant and cannot change the ranking; its apparent benefit comes from acting as a time fixed effect during training. Pooling the four folds' predictions into one ranking tells the real story:

| Feature set | n | LR within-month | LR pooled | LightGBM within-month | LightGBM pooled | LightGBM Brier |
|---|---:|---:|---:|---:|---:|---:|
| full | 57 | 0.167 | 0.123 | 0.161 | 0.146 | 0.0929 |
| − month | 56 | 0.147 | 0.155 | 0.157 | 0.150 | 0.0927 |
| − month − redundant | 49 | 0.148 | 0.156 | 0.156 | 0.152 | 0.0922 |
| **− month − product** | **47** | 0.147 | 0.155 | **0.160** | **0.159** | **0.0921** |
| − month − product − redundant | 42 | 0.148 | 0.156 | 0.154 | 0.150 | 0.0922 |

**Decision.** Drop `purchase_month` (it memorises one-off months and wrecks cross-month ranking) and the 9 product features, leaving 47.

## Stationarity audit: clock features (added after a first look at the test set; disclosed)

A first evaluation of the 47-feature models on the test period showed every tuned model losing to the 5-feature baseline (kept in `reports/results/07_first_pass_test_results.csv`). The diagnosis used **no test labels**:

| Check | Result |
|---|---|
| Spearman ρ with purchase date (training period) | `market_resolved_30d` **0.95**, `route_resolved_n_min` 0.43, `market_late_rate_30d` 0.42, `seller_resolved_n_min` 0.32, `seller_prior_orders_min` 0.31 |
| Adversarial validation (train-period vs validation rows), 47 features | AUC **1.000**. 59% of the gain from `market_resolved_30d`, 22% from `market_late_rate_30d` |
| Same, after removing the 5 clocks | Still AUC 1.000, but now separated by 90-day destination/seller **late rates** and holiday counts. That is genuine drift in order-level risk, not a clock |

| Learner | Feature set | Within-month | Pooled | Validation |
|---|---|---:|---:|---:|
| Logistic regression | 47 | 0.147 | 0.155 | 0.224 |
| | **42 (clocks removed)** | **0.152** | **0.160** | **0.234** |
| LightGBM | 47 | 0.160 | 0.159 | 0.196 |
| | 42 (clocks removed) | 0.141 | 0.148 | 0.188 |

The evidence is mixed: logistic regression gains, LightGBM loses. Inside the training window, volume and lateness rose together, so a clock still "works" there; it only fails when the trend reverses.

**Decision.** Remove the five clock features: two market-level aggregates and three cumulative counts that grow with the marketplace's age. The principle is that a feature should describe the order, not the calendar. **42 features remain**: geography 10, basket 9, history 9, calendar 6, promise 4, payment 4. All drops and reasons are in `olist.features.DROPPED_AFTER_ABLATION`.
