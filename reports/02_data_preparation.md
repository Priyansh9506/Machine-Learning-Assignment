# Stage 3 report: order-level dataset and leakage controls

Evidence: [`notebooks/02_data_preparation.ipynb`](../notebooks/02_data_preparation.ipynb) (executed; about 30 s, peak RSS about 660 MB).
Output: `data/processed/orders_model.parquet`, with 98,666 orders × 74 columns (20.5 MB) and 57 registered features.

## What was built

| Module | Role |
|---|---|
| `src/olist/dataset.py` | Builds the labels (`late_day`, `late_ts`, `promise_failed`), the outcome-known time `known_at`, the population flags and the data-quality flags. It aggregates items, products, sellers and payments to order grain and assembles the dataset. |
| `src/olist/history.py` | Point-in-time features computed with `merge_asof` over cumulative sums. Each order sees only outcomes with `known_at` < its purchase time. |
| `src/olist/features.py` | Registry of the 57 allowed inputs (group, source, when known), the `FORBIDDEN` columns, and the leakage-audit table. |
| `src/olist/split.py` | Chronological split and the maturity purge. |
| `src/olist/calendar.py` | Brazilian holidays 2016–2019 and Black Friday dates, hard-coded. |

**Join validation.** Every order-level join keeps 99,441 rows. The only unmatched rows are 775 orders without items, which are never delivered and are dropped, and one order without a payment record. Item-level joins keep 112,650 rows.

## Feature groups (57 features)

| Group | Count | Features |
|---|---|---|
| promise | 4 | promised days, business days, holidays in the window, promise weekday |
| calendar | 7 | hour, weekday, month, day of month, weekend, Black Friday period, holiday within 7 days |
| basket | 9 | items, products, sellers, categories, value, freight, freight ratio, zero-freight flag |
| product | 9 | weight, volume (total and max), photos, text lengths, main category, missing-category flag |
| geography | 10 | customer state, main seller state, distance (max/min), customer lat/lng, same-state, imputed-coordinate flags |
| payment | 4 | main type, number of types, installments, voucher (ablation-gated) |
| history | 14 | see below |

The 14 history features cover:
- **Seller:** late rate (all-time and 90-day, smoothed), resolved count, cold start, prior and recent order volume.
- **Route:** late rate and resolved count.
- **Destination state:** late rate (all-time and 90-day).
- **Market:** 30-day late rate and resolved volume.
- **Customer:** prior orders.

## Population

| | Orders | Late | Rate |
|---|---:|---:|---:|
| Primary (`eligible_primary`) | 89,795 | 6,138 | 6.84% |
| Sensitivity (`promise_failed`, adds the 1,624 never-delivered, non-cancelled orders in the window) | 91,419 | 7,762 | 8.49% |

Orders with impossible post-checkout timestamps are flagged, not dropped. Inside the population there are 115 carrier-before-purchase cases, 1,261 carrier-before-approval cases and 19 delivered-before-carrier cases.

## Leakage controls and evidence

1. **`known_at = min(delivered_ts, promised_date + 1 day)`.** This is when the marketplace learns the outcome. History counts events strictly before the purchase time.
2. **Tests** (`tests/test_history.py`, all passing):
   - toy cases for strictness and ties;
   - a 400-query brute-force comparison on random data;
   - a brute-force check of 150 real seller features;
   - a **future-perturbation test**: randomising every outcome known after 2018-01-01 leaves the features of all earlier orders bit-identical.
3. **Deliberately leaky comparison** (training orders only, single-feature scorer of `late_day`):

| Seller late rate | ROC-AUC | PR-AUC |
|---|---:|---:|
| Leaky (full data, includes own and future orders) | 0.709 | 0.180 |
| Naive (orders *delivered* before purchase) | 0.554 | 0.082 |
| As-of (`known_at`, used) | 0.552 | 0.081 |

The leaky version inflates performance a great deal. The as-of and naive rules rank orders almost identically: the level bias identified in Stage 1 exists (+0.011), but it is the same for late and on-time orders. **Seller history alone is weak.** Its value has to come from combinations with other features, which the Stage 5 ablations will test.

## Split and purge (correction made during this stage)

| Split | Purchases | Orders | Late rate | After maturity purge |
|---|---|---:|---:|---:|
| train | 2017-01 → 2018-02 | 57,050 | 6.63% | 50,356 kept |
| val | 2018-03 → 2018-04 | 13,801 | 11.84% | – |
| test | 2018-05 → 2018-07 | 18,944 | 3.81% | – |

- **Purge rule.** Purging training rows on `known_at < cutoff` was rejected. It keeps recent orders only if they were delivered early, which removes a sample that is mostly late (26.6% late rate) and biases the labels toward "on time".
- **Rule used.** The **maturity rule** keeps a row only if `promised date + 1 day ≤ cutoff`, which is known at checkout. Its purged rows have a 14.06% late rate, the same as February 2018 as a whole (14.13%). It removes a period, not an outcome.
- **Base-rate shift.** The late rate is 11.8% in validation against 3.8% in test. Calibration and thresholds chosen on validation may not transfer. Stage 4/5 should consider walk-forward validation folds and report per-month results.
- **Customer overlap.** 313 customers appear in both train and test, which is negligible.

## Open questions for Stage 4 (EDA)

- Whether to keep `purchase_month`. With only about 1.5 years of data, a month's effect is learned from a single occurrence, so it risks memorisation.
- Imputation for weight and volume, and for the product text statistics (1.45% missing, all tied to missing category).
- `market_late_rate_30d` lags real drift (for example, the March 2018 spike carries into April). Its usefulness needs testing, not assuming.
- The split dates, given the validation/test base-rate gap.
