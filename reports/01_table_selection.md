# Stage 1–2 report: data audit and table selection

Evidence: [`notebooks/01_data_audit.ipynb`](../notebooks/01_data_audit.ipynb) (executed; about 16 s, peak RSS about 360 MB).
All counts below were computed on the supplied files. Figures quoted from the proposal are treated as prior estimates and checked in section 11 of the notebook.

## 1. Findings that change the original design

1. **Timestamps are day-first and minute-precision.** `orders` and `order_items` store `DD-MM-YYYY HH:MM` with no seconds, which differs from the public Kaggle release. Loaders therefore parse with an explicit format and fail on any mismatch (`olist.io.parse_dates`, covered by tests).
2. **The "full-timestamp" sensitivity rule is really an on-day rule.** All 99,441 estimated delivery dates fall at 00:00. So `delivered > estimated` only differs from the day rule for the **1,292** orders delivered *on* the promised day. The primary label is `late_day` (6,534 / 96,470 = 6.77%). `late_ts` (7,826 = 8.11%) is kept as a strict-promise sensitivity check.
3. **Right-censoring has a measurable cutoff.** Recorded deliveries fall from 1,607 in the week of 27 Aug 2018 to 28 in the week of 3 Sep. The late rate per purchase week also falls monotonically through August (≈9% → 8.5% → 3.6% → 2.2% → 0%). This means slow August orders are missing from the data. The effective observation cutoff is `T_OBS = 2018-09-03`.
4. **Some excluded orders are known failures.** 1,729 orders were never delivered and never cancelled (1,107 `shipped`, 314 `invoiced`, 301 `processing`, …). Every one of them is past its promised date. A "delivered only" population silently drops them, so they become the positive class of a `promise_failed` sensitivity label.
5. **The impossible timestamps involve post-checkout fields only.** The counts are 161 carrier handoffs before purchase, 1,358 carrier handoffs before approval, and 23 deliveries before carrier handoff. None of these fields is a Task A input, and the fields that define the label have no impossible orderings. The affected orders are therefore **flagged, not dropped**, for Task A.
6. **The review timing figure was an artefact.** The proposal reported 8,320 reviews written before delivery. That count compares date-only review values with delivery *timestamps*. At day level the figure is **5,126**.
7. **`shipping_limit_date` is set after checkout.** It falls at roughly approval + 6 days, so it is excluded from Task A features.
8. **Missing coordinates can be recovered at a fine level.** 279 customers and 7 sellers have zip prefixes missing from geolocation. Of these, 276 customers and all 7 sellers are recovered with a 3-digit-prefix centroid, and 3 customers fall back to a state centroid. This is much finer than the state centroid the proposal planned.
9. **Seller history must use the time a label became known.** The proposal used "orders delivered before this purchase". That under-counts late orders, because late orders resolve later. The Stage 3 rule is `known_at = delivered date if on time, else estimated date + 1 day`.

## 2. Table decisions

| Table | Grain / key | Class | Decision and reason |
|---|---|---|---|
| `orders` | order / `order_id` | **Essential** | Holds the label (delivered vs estimated date), the prediction anchor (purchase time), the promise (estimated date, which is known at checkout) and the population filter (status). `approved_at` and `delivered_carrier_date` are loaded only for anomaly flags and the optional Task B. |
| `order_items` | item / `order_id + order_item_id` | **Essential** | Provides the seller link, price, freight, item count and seller count (1,278 multi-seller orders). It is 1:m with orders and is **aggregated to order grain before joining**. `shipping_limit_date` is excluded (post-checkout). |
| `customers` | order / `customer_id` (1:1 with orders) | **Essential** | Destination zip and state. It joins 1:1 with 100% coverage. |
| `sellers` | seller / `seller_id` | **Essential** | Origin zip and state for distance and route features. It is tiny (3,095 rows), and m:1 into items has 100% coverage. |
| `products` | product / `product_id` | **Useful** | Weight, volume, category and listing quality, aggregated over items. Cheap (33k rows). Gaps: 610 missing category, 2 missing dimensions, 4 with zero weight; these are imputed or flagged inside training folds. |
| `geolocation` | none (1,000,163 rows, 261,831 exact duplicates, 42 outside Brazil) | **Useful, via compact map only** | Reduced once to a 19,010-row zip→(lat, lng) median map, cached as `data/interim/zip_centroids.parquet`, with a `geo_source` flag. The raw table is never joined: a naive zip join to customers alone would produce about 18.2M rows. |
| `order_payments` | payment / `order_id + payment_sequential` | **Useful, ablation-gated** | Payment type and installments are chosen at checkout. Boleto has a median approval lag of 29 h vs 0.3 h for cards. Aggregated to one row per order (main type, max installments, number of methods). It stays only if the ablation shows value. `payment_value` is redundant with price + freight. |
| `order_reviews` | review × order (`review_id` reused across 789 orders) | **Exclude from Task A** | Every column exists only after delivery, so using any of it would leak the outcome. It is used only in the secondary association analysis (bad-review rate 62.4% for late orders vs 9.3% for on-time; no causal claim) and in the optional Task B. |
| `category_translation` | category | **Optional (presentation)** | Gives English labels for figures and the report. It is not a model input, because the Portuguese code carries the same information. Two missing translations are supplied manually in `config.EXTRA_TRANSLATIONS`. |

## 3. Integration plan (Stage 3)

```
items ──m:1── products
items ──m:1── sellers ──m:1── zip_map         (seller coordinates, route)
items ─► aggregate to order grain (counts, sums, max distance, n_sellers, ...)
orders ──1:1── customers ──m:1── zip_map      (customer coordinates)
orders ──1:1── items_agg
orders ──1:1── payments_agg                    (ablation-gated)
```

- Every merge goes through `olist.joins.safe_merge`. It accepts only `1:1` or `m:1`, asserts that the row count is unchanged, and logs unmatched keys.
- Seller and route history features are computed as of each order's purchase time from `known_at`. They are Bayesian-smoothed and come with a prior-count feature.
- Pre-2017 orders contribute to history but are not training rows.

## 4. Analytical population (provisional)

| Step | Orders | Late (day) | Late % |
|---|---:|---:|---:|
| All orders | 99,441 | – | – |
| Delivered with delivery date | 96,470 | 6,534 | 6.77 |
| Purchase ≥ 2017-01-01 (2016 sparse) | 96,203 | 6,531 | 6.79 |
| Purchase < 2018-08-01 (August censored) | 89,852 | 6,138 | 6.83 |
| Promised date ≤ `T_OBS` | **89,795** | **6,138** | **6.84** |

The provisional chronological split is train 2017-01 → 2018-02, validation 2018-03 → 04 and test 2018-05 → 07. Training rows whose promised day had not passed by the training cutoff are purged (the maturity rule; see `reports/02_data_preparation.md`). The split will be confirmed after the EDA in Stage 4.

## 5. Not yet decided (Stage 3 onwards)

- Final split dates and the walk-forward fold layout.
- Feature list, encodings and imputation (all fitted on training data only).
- Whether Task B is built as a model or stays an association analysis. It is deferred until Task A is complete.
