# Olist Delivery Promise Risk: ML Topic Proposal

Oct 12, 2026 · @Priyansh Patel

## Proposed topic

**Delivery Promise Risk: Predicting Late Deliveries and Customer Dissatisfaction in E-Commerce using Classical Machine Learning (Olist Brazilian E-Commerce Dataset)**

We propose two linked binary classification tasks on the public Olist dataset of about 99,000 real marketplace orders (2016 to 2018). **Task A (primary)** predicts, at the moment a customer places an order, whether it will be delivered after the promised date. **Task B (secondary)** predicts, at the moment an order is delivered, whether the customer will leave a dissatisfied review (1 or 2 stars). Late delivery is the main driver of bad reviews in this data, so the two tasks form one story: catch the risk early, and catch the unhappy customer before the review is written.

The project is pure ML, built in Python notebooks plus a small reusable package, with no web UI. It covers all 12 stages of the guideline: data cleaning across 9 joined tables, preprocessing, imbalance handling, analytical EDA, feature engineering, a leakage audit, a time-based split, baselines, classical models (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting), hyperparameter tuning, evaluation with explainability, and a cost-sensitive decision analysis. All figures quoted below are counts from public copies of the data that the team will recompute on the official download before reporting.

## Problem definition and use case

Online marketplaces promise a delivery date at checkout. When that promise breaks, customers leave poor reviews and are less likely to return. In Olist, late orders average 2.57 review stars against 4.31 for on-time orders ([source](https://github.com/EyadMohamed07/Olist-Business-Discovery-The-Delivery-Promise-Gap)). The goal is to flag risky orders early enough to act.

|  | Task A: late-delivery risk (primary) | Task B: dissatisfaction risk (secondary) |
| --- | --- | --- |
| Question | Will this order arrive after the promised date? | Will this customer give 1 or 2 stars? |
| Prediction moment | At checkout (order purchase timestamp) | At delivery confirmation, before the review request |
| Inputs | Only what is known at checkout: products, price, freight, seller, customer location, promised date, calendar | Task A inputs plus delivery facts: actual delivery days, days late or early, carrier handoff delay |
| Output | Probability of late delivery plus a recommended action | Probability of a bad review plus a recommended action |
| Stakeholder | Marketplace operations manager | Customer support team |
| Action enabled | Expedite shipping, give a more honest ETA, or offer a voucher | Proactive apology or support call to the top-k riskiest customers |
| Constraint | Expediting costs money, so it cannot be applied to every order | Support team can only contact a limited number of customers per day |
| Target definition | delivered date (day level) later than estimated delivery date | review\_score of 1 or 2 |

**Success criteria.** ML terms: PR-AUC, recall and F1 on the minority class, measured on an out-of-time test period. Business terms: expected cost saved per 1,000 orders against two simple policies, "do nothing" and "act on every order".

## Dataset

The data is the [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce), real anonymised orders from a Brazilian marketplace, released under **CC BY-NC-SA 4.0** (non-commercial, share-alike), which permits academic use with attribution.

| Item | Detail |
| --- | --- |
| Source and owner | Olist, published on Kaggle (version 2); also mirrored on Olist's own [GitHub](https://github.com/olist/work-at-olist-data) |
| Tables | 9 CSVs: orders, order\_items, order\_payments, order\_reviews, customers, sellers, products, geolocation, category name translation |
| Size | 99,441 orders, 32,951 products, 96,096 unique customers, 1,000,163 geolocation rows |
| Time range | Purchases from Sep 2016 to Oct 2018 (thin months at both ends) |
| Data types | Timestamps, numeric (price, freight, weight, dimensions), categorical (state, city, category, payment type), text (Portuguese review comments), geographic (zip prefix, lat/long) |
| Task A target | Late delivery among 96,470 delivered orders: 6,534 late (6.77%) at day level, or 7,826 (8.11%) if full timestamps are compared |
| Task B target | Share of 1 to 2 star reviews: to be computed on day 1 |
| Known limitations | One marketplace, one country, 2016 to 2018 only; review behaviour may not match Indian markets; no information on the actual carrier used |

We choose the day-level late rule as the main label because the customer is promised a date, not a time. The timestamp rule is reported as a sensitivity check.

## Data preparation and preprocessing

The raw data needs real cleaning work, which is why it suits this assignment. Every step below will be logged with row counts before and after, so the report shows a full record-count funnel.

### Data preparation (integration and cleaning)

1. **Join the 9 tables into one order-level table.** orders to order\_items (one to many), items to products and sellers, orders to customers, customers and sellers to geolocation by zip prefix, orders to reviews and payments. Cardinality and orphan counts are checked at every join.
2. **Aggregate multi-item orders.** Orders can contain up to 21 items, and 1,278 orders involve more than one seller, so item-level fields become order-level features (item count, total weight, max seller distance, number of sellers).
3. **Deduplicate geolocation.** 261,831 of its 1,000,163 rows are exact duplicates. We collapse it to one median latitude and longitude per zip prefix and drop 42 points that fall outside Brazil.
4. **Resolve review duplicates.** 547 orders have more than one review, and 768 orders have none. We keep the latest review per order for Task B and exclude orders with no review from Task B only.
5. **Fix impossible timestamps.** 166 orders show the carrier handoff before the purchase, and 23 show delivery before the carrier handoff. These are flagged and removed, with the count reported.
6. **Define the modelling population.** Task A uses delivered orders only. Orders purchased in the last weeks of the data are dropped because many had not been delivered yet, so their outcome is unknown (right-censoring).
7. **Fix formats.** Parse all timestamps, rename the misspelled source columns (product\_name\_lenght and similar), and standardise city names (case and accents).

### Preprocessing decisions

| Issue | Where | Treatment | Why |
| --- | --- | --- | --- |
| Missing category | 610 products | New "unknown" category | Missing is informative; dropping loses orders |
| Untranslated category | 2 of 73 categories | Translate manually | The translation table covers only 71 |
| Missing weight or dimensions | A few products | Median within category, fitted on training data | Small share; category median is a sensible prior |
| Missing coordinates | 279 customer and 8 seller zip prefixes | State centroid plus a "missing location" flag | Keeps the order and tells the model it was imputed |
| Zero weight or zero freight | 4 products, 383 item rows | Keep, add a flag | Likely real promotions or data entry; examined in EDA |
| Heavy tails | Price up to 6,735 BRL, freight up to 409.68 BRL | log1p transform; winsorise at 1st and 99th percentile for linear models only | Trees are robust to outliers, linear models are not |
| Low-cardinality categoricals | Customer state (27), payment type | One-hot encoding | Few levels, no order |
| High-cardinality categoricals | Product category (73), seller id, city | Frequency or target encoding, fitted inside each training fold | One-hot would explode; target encoding outside folds leaks |
| Scaling | Numeric features | StandardScaler for Logistic Regression and kNN; none for tree models | Distance and gradient based models need it, trees do not |

All imputers, encoders and scalers live inside one scikit-learn Pipeline with a ColumnTransformer, fitted only on training data and applied unchanged to validation and test data.

## Balancing, EDA, features, leakage and split

### Class imbalance

Task A is imbalanced at about 6.8% late, and Task B will be measured on day 1. Plain accuracy would reach about 93% by predicting "on time" for every order, so it cannot be the main metric. We will compare five strategies on validation PR-AUC:

1. No handling (reference)
2. Class weights (class\_weight="balanced", scale\_pos\_weight for boosting)
3. Random undersampling of the majority class
4. SMOTE-NC (the SMOTE variant for mixed numeric and categorical data), run inside an imblearn Pipeline so it only touches training folds
5. Threshold moving on an unweighted model

The validation and test sets always keep their natural class ratio. We expect class weights plus threshold tuning to be enough, but the comparison decides.

### EDA: questions that drive decisions

| Question | Analysis | Decision it informs |
| --- | --- | --- |
| Does the late rate change over time? | Monthly late rate line plot (public counts show 1.4% to 21.4%) | Time-based split and per-month evaluation |
| Does distance cause lateness? | Late rate by distance decile and by customer state | Distance and same-state features |
| Do some sellers fail repeatedly? | Distribution of seller late rates | Seller history feature |
| Are generous promises safer? | Late rate against promised days | Promised-days feature |
| How much does lateness hurt reviews? | Review score split by late and on-time | Task B motivation and cost assumptions |
| Which numeric fields have outliers? | Box plots of price, freight, weight | Log transform and winsorising |
| Which features are redundant? | Spearman correlation, Cramér's V for categoricals | Drop or merge redundant features |
| Is missingness random? | Missingness heatmap by month and category | Imputation plus missing flags |

Every plot in the report will carry three lines: the question, the finding, and the decision it changed.

### Feature engineering

| Feature | Definition | Why it should help | Task |
| --- | --- | --- | --- |
| distance\_km | Haversine distance from seller to customer (max over sellers) | Longer routes take longer | A, B |
| same\_state | Seller and customer in the same state | Intra-state delivery is simpler | A, B |
| promised\_days | Estimated delivery date minus purchase date | Tight promises break more often | A, B |
| freight\_ratio | Freight value divided by price | Expensive shipping hints at hard routes | A, B |
| total\_weight\_g, volume\_cm3 | Summed over items | Heavy, bulky parcels move slower | A, B |
| n\_items, n\_sellers | Count per order | Multi-seller orders need consolidation | A, B |
| purchase weekday, month, hour | From purchase timestamp | Weekend and seasonal load | A, B |
| holiday and Black Friday flags | Brazilian holiday calendar | Peak periods overload carriers | A, B |
| seller\_prior\_late\_rate | Seller's late rate using only orders delivered before this purchase | Past reliability predicts future reliability | A, B |
| seller\_prior\_volume, new\_seller | Orders the seller completed before this purchase | Cold-start sellers are riskier | A, B |
| route\_prior\_late\_rate | Late rate for the seller-state to customer-state route, same as-of rule | Some corridors are slow | A, B |
| actual\_delivery\_days, days\_late | Delivered date minus purchase, delivered minus promised | The direct cause of a bad review | B only |
| carrier\_delay\_days | Carrier handoff minus approval | Slow dispatch frustrates customers | B only |

Feature selection will use a correlation filter, L1-regularised Logistic Regression, permutation importance, and an ablation table that removes one feature group at a time.

### Leakage audit

The rule is simple: a feature is allowed only if it is known at the prediction moment.

| Column | Known when | Task A | Task B |
| --- | --- | --- | --- |
| order\_delivered\_customer\_date | At delivery | Drop | Use (derived features) |
| order\_delivered\_carrier\_date | At dispatch | Drop | Use |
| order\_approved\_at | After payment approval | Drop | Use |
| order\_status | Final state | Drop | Drop |
| review\_score, comments, review dates | After delivery | Drop | Target only |
| order\_estimated\_delivery\_date | Shown at checkout | Keep as promised\_days, with and without it tested | Keep |
| Seller late rate over full data | Uses future orders | Replace with the as-of version | Replace with the as-of version |

Two extra checks. First, 8,320 reviews in the public data were created before the recorded delivery time; for Task B these are analysed separately, since those customers reviewed before the delivery facts existed. Second, a unit test fails if any feature for an order uses information dated after that order's prediction moment. We will also show one deliberately leaky model to prove how much the audit matters.

### Split strategy

&#91;embedded content: time-based split · train, validation, test\]

A random split would mix future orders into training and hide the strong month-to-month drift. Hyperparameters are tuned with walk-forward cross-validation inside the training period, the threshold is chosen on validation, and the test period is used once. Only 2,997 of 96,096 customers ordered more than once, so customer overlap across periods is small, but it will be counted and reported. Random seeds are fixed at 42 everywhere.

## Baselines, models and hyperparameter tuning

### Baselines (stage 8)

Every model must beat three simple references on the same split and metrics:

1. **Majority class:** always predict "on time" (or "satisfied"). High accuracy, zero recall. It shows why accuracy is the wrong metric.
2. **Business rule:** flag an order as risky if the seller's prior late rate or the distance is above a cut-off chosen on training data. This is what an ops team could do without ML.
3. **Simple Logistic Regression** on five core features (distance, promised days, seller prior late rate, freight ratio, month).

### Classical models compared (stage 9)

| Model | Why it suits this problem | Assumptions and weaknesses | Needs scaling |
| --- | --- | --- | --- |
| Logistic Regression (L1 or L2) | Interpretable coefficients, well-calibrated probabilities | Assumes a linear relationship in log-odds; misses interactions | Yes |
| Decision Tree (CART) | Produces readable if-then rules an ops manager can follow | Overfits and is unstable without depth and leaf limits | No |
| Random Forest | Averages many trees, captures interactions, robust to outliers | Slower, larger, probabilities less calibrated | No |
| Gradient Boosting (XGBoost or LightGBM) | Usually the strongest model family on tabular data | Many hyperparameters, overfits if not tuned with early stopping | No |
| k-Nearest Neighbours (optional contrast) | Non-parametric reference point | Slow on 90k rows, sensitive to scaling and irrelevant features | Yes |

This gives four genuinely different families: linear, single tree, bagged trees and boosted trees. The Decision Tree is kept even if it scores lower, because its rules double as an explanation for the business.

### Hyperparameter tuning (stage 10)

| Model | Hyperparameters | Search range | Method |
| --- | --- | --- | --- |
| Logistic Regression | C; penalty; l1\_ratio; class\_weight | C from 0.001 to 100 (log scale); l1, l2, elasticnet; 0 to 1; none or balanced | Randomised search, 50 trials |
| Decision Tree | max\_depth; min\_samples\_leaf; min\_samples\_split; criterion; ccp\_alpha; class\_weight | 3 to 20; 20 to 1,000; 2 to 200; gini or entropy; 0 to 0.01; none or balanced | Grid search, plus a depth-vs-score curve to show overfitting |
| Random Forest | n\_estimators; max\_depth; max\_features; min\_samples\_leaf; class\_weight | 200 to 800; 6 to 30 or none; sqrt, 0.3, 0.5; 5 to 200; balanced\_subsample | Randomised search, 40 trials |
| XGBoost or LightGBM | learning\_rate; max\_depth or num\_leaves; min\_child\_weight; subsample; colsample\_bytree; reg\_lambda; scale\_pos\_weight | 0.01 to 0.2 (log); 3 to 10 or 15 to 255; 1 to 50; 0.6 to 1; 0.6 to 1; 0 to 10; 1 to the class ratio | Optuna (Bayesian, TPE), 100 trials, early stopping on number of trees |

All searches use walk-forward cross-validation inside the training period (about four monthly folds) and score on PR-AUC (average precision). Every trial is logged, and the report shows the search space, a score-versus-trial plot, and the best configuration. Train, cross-validation and test scores are reported side by side so any overfitting gap is visible.

## Evaluation, explainability and decision analysis

### Metrics (stage 11)

| Metric | Why we use it |
| --- | --- |
| PR-AUC (primary) | Focuses on the rare late class; not inflated by the many on-time orders |
| Recall, precision, F1 on the minority class | Recall = share of late orders caught; precision = share of alerts that were real |
| Balanced accuracy | Average of recall on both classes, fair under imbalance |
| ROC-AUC (secondary) | For comparison with published work |
| Confusion matrix at the chosen threshold | Shows exactly what each error type costs |
| Brier score and calibration curve | Probabilities feed the cost decision, so they must be trustworthy |
| Precision@k (Task B) | Support can contact only k customers a day |
| PR-AUC per test month | Shows how performance changes as the late rate drifts |

Every model gets one results table with train, cross-validation (mean and standard deviation) and test scores. The top two models are compared with a paired bootstrap 95% confidence interval on PR-AUC, so "model B is better" is backed by more than a single number.

### Explainability and error analysis

- **Global:** Logistic Regression coefficients, the Decision Tree's top rules drawn as a tree, permutation importance, and a SHAP summary plot for the boosting model.
- **Local:** SHAP explanations for three real test orders (a correct alert, a missed late order, a false alarm), written as plain reasons such as "seller 1,800 km away with a slow history".
- **Error slices:** recall and precision by customer region, holiday and Black Friday periods, new sellers, multi-seller orders and heavy parcels, to find where the model fails.
- **Failure cases:** the 20 most confident wrong predictions are inspected by hand and explained.

### Cost-sensitive decision layer

A high score is not the end goal; a better decision is. Each order gets an action only when the expected benefit beats the expected cost. For Task A, acting (expediting) is worth it when:

```latex
p \cdot B \;>\; (1 - p) \cdot C \quad\Longrightarrow\quad t^{*} = \frac{C}{C + B}
```

Here p is the calibrated late probability, B is the benefit of acting on a truly late order (damage avoided), and C is the cost of acting on an order that would have been on time (wasted expedite cost). B and C are anchored to each order's real price and freight values, for example an expedite cost of 0.3 to 1.0 times the freight value.

The report will show:

1. A profit curve (expected saving against threshold) for the model, "do nothing", "act on every order" and the business-rule baseline.
2. Probability calibration (Platt or isotonic, fitted on validation only) before costs are applied.
3. A sensitivity (tornado) chart that varies every assumed cost by plus or minus 50% and checks whether the recommended policy changes.
4. A drift experiment: a model trained once versus a model retrained or recalibrated monthly, evaluated month by month on the test period.

For Task B the same logic becomes a top-k ranking: contact the k customers most likely to be unhappy and report how many bad reviews that would catch.

## Impact, limitations and deliverables

### Use case and impact (stage 12)

- **Who benefits:** marketplace operations (fewer broken promises), customer support (targeted outreach instead of reacting to bad reviews), and customers (honest delivery dates).
- **How it is used:** the trained pipeline scores each new order at checkout and each delivered order before the review request, then returns a probability and a recommended action.
- **New or unseen data:** a predict.py script loads the saved pipeline and scores raw orders it has never seen, applying the same cleaning, features and threshold.
- **Expected impact:** reported as expected cost saved per 1,000 orders against "do nothing". This is an estimate under stated assumptions, not a measured business result.

### Limitations and risks

- One Brazilian marketplace from 2016 to 2018; results may not transfer to other countries or years.
- Costs of expediting and of a bad review are assumptions, which is why the sensitivity analysis is part of the core work.
- The data is observational, so we cannot prove that expediting actually prevents lateness; that would need an experiment.
- Strong monthly drift means a fixed threshold can go stale; this is measured, not hidden.
- The licence is non-commercial, which is fine for coursework but not for a commercial product.
- Future work: uplift modelling with experimental data, Portuguese review text as an NLP feature, and testing on Indian e-commerce logistics data.

### Deliverables (no web UI)

| Deliverable | Content |
| --- | --- |
| Notebooks 01 to 08 | Data card and preparation, preprocessing, EDA, features and leakage audit, baselines, modelling and tuning, evaluation and explainability, decision analysis |
| src package | Reusable modules for data loading, features, models, evaluation and the decision logic |
| tests folder | Unit tests for feature functions (including the no-future-data test) and an integration test for the full pipeline |
| predict.py | Scores new, unseen raw orders with the saved pipeline |
| Report | Stage-applicability table, decision log, results tables, figures, limitations |
| SOURCES and AI-use log | Dataset licence and citation, libraries, code references, and where AI tools assisted and how the output was checked |

Reproducibility: pinned requirements file, fixed seeds, a checksum of the downloaded data, and one command to rerun the pipeline from raw files.

### Questions for faculty

- Is a public Kaggle dataset acceptable for this assignment?
- Should we submit both tasks, or Task A only with Task B as an extension?
- Is a notebook plus prediction script acceptable instead of a deployed web application?
- What are the marks weighting, team size, report format and deadline?

### Sources

- [Olist Brazilian E-Commerce Public Dataset (Kaggle)](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)
- [Olist dataset mirror (GitHub)](https://github.com/olist/work-at-olist-data)
- [Olist delivery risk dataset card, leakage list and late rate (Hugging Face)](https://huggingface.co/datasets/samuelalex37/olist-delivery-risk-data)
- [Review scores of late vs on-time orders (GitHub analysis)](https://github.com/EyadMohamed07/Olist-Business-Discovery-The-Delivery-Promise-Gap)
- Machine Learning Innovative Assignment: IA Guidelines and Requirements (course document)
