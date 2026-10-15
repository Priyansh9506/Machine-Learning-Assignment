# Stage 4 report: exploratory data analysis

Evidence: [`notebooks/03_eda.ipynb`](../notebooks/03_eda.ipynb) (executed; about 15 s, peak RSS about 650 MB). Figures: `reports/figures/03_*.png`.

**Rule.** Every statistic was computed on the **training period only** (purchases 2017-01 → 2018-02, 57,050 orders, 6.63% late). Validation and test months were not examined, so no modelling decision was tuned on data later used to judge the model.

## Questions, findings and decisions

| # | Question | Finding | Decision |
|---|---|---|---|
| 1 | Does the late rate drift? | Monthly rate swings 2.8% → 14.1% inside training. The as-of `market_late_rate_30d` barely tracks it (weekly correlation 0.03) and lags spikes by about 6 weeks, because late outcomes are only known after the promised date. | Chronological split, per-month reporting, market features sent to the ablation, drift experiment in Stage 12 |
| 2 | Does distance matter? | Late rate 3.6% (nearest decile, 18 km) → 10.7% (farthest, about 2,100 km). Cross-state 7.9% vs same-state 4.1%. By destination: RO 2.4%, MG 3.4% … MA 16.4%, AL 24.1%. | Keep distance, same-state flag and state; log1p for linear models |
| 3 | Do sellers differ? | Yes: overdispersion χ² = 1,403 on 405 df (p ≈ 4e-110). But the as-of seller rate alone is weak (ROC-AUC 0.552, PR-AUC 0.081 against a 0.066 base rate). Route and destination-state rates are stronger (ROC-AUC 0.63–0.64). | Keep seller history plus the cold-start flag, and let the ablation measure it |
| 4 | Are generous promises safer? | Not monotonic: ≤10 days 11.4%, 20–30 days 7.5%, >40 days 3.1%. The median promise (25 days) is more than double the median delivery (11 days). Within a distance band, longer promises are safer (farthest quintile: 15.7% at P2 vs 6.4% at P5). | Keep both promise features; include trees for the distance × promise interaction |
| 5 | Does lateness hurt reviews? | Late orders average 2.18★ vs 4.27★. Bad-review share is 65.2% vs 9.5%. It is flat at about 10% for early deliveries and jumps to about 80% a week late. | Confirms the business case; sets the damage scale in the cost model; `days_late` for Task B |
| 6 | Heavy tails? | `total_price` skew 11.5 (max 13,440 vs p99 969 BRL). Freight, weight and volume skew 6–9. | log1p + winsorise at 1/99% in the linear pipeline only |
| 7 | Redundancy? | 12 pairs with \|ρ\| ≥ 0.85 (e.g. max/min distance 0.997, promised/business days 0.973). Categoricals are nearly independent (Cramér's V ≤ 0.22). | Near-duplicates go to the ablation as a group |
| 8 | Missingness? | 7 features, 16 orders (weight/volume) and 1,025 orders (text/photos, all tied to a missing category). Even across months; no signal (6.73% vs 6.63%). | Median + missing indicator for linear models; native NaN for trees |
| 9 | Calendar? | Weekday effect small (5.8–7.2%). Black Friday window 16.8% vs 5.8%. **The same month differs across years** (Feb 2017 3.0% vs Feb 2018 14.1%). | Drop `purchase_month` (confirmed in Stage 5); keep Black Friday and holiday flags |
| 10 | Single-feature signal? | Best features are geographic; no feature exceeds folded ROC-AUC 0.66. | Group ablations, not univariate filters |

## Key figures

| Figure | Shows |
|---|---|
| `03_late_rate_by_split.png` | Monthly late rate coloured by split; validation and test sit at opposite extremes |
| `03_market_feature_lag.png` | The as-of market feature trailing the real weekly rate |
| `03_distance_and_state.png` | Late rate by distance decile and by destination state |
| `03_seller_overdispersion.png` | Observed seller late rates vs what chance alone would give |
| `03_promises.png` | Promise length, promised vs actual days, distance × promise heatmap |
| `03_reviews_vs_lateness.png` | Review distribution and the jump at day 0 |
| `03_skew_boxplots.png`, `03_spearman_heatmap.png`, `03_missingness.png`, `03_calendar.png`, `03_univariate_signal.png` | Questions 6–10 |
