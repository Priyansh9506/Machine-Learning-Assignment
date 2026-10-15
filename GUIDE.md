# 📘 Project Guide: Olist Delivery Promise Risk

*A plain-language walkthrough of the whole project: what we built, why, what we found, and how to explain it.*

**Contents:** [1. In one minute](#1-the-project-in-one-minute) · [2. The story, stage by stage](#2-the-story-stage-by-stage) · [3. Ideas in plain words](#3-the-ideas-you-need-in-plain-words) · [4. How to run it](#4-how-to-run-it) · [5. Where everything lives](#5-where-everything-lives) · [6. What went wrong, and how we fixed it](#6-what-went-wrong-and-how-we-fixed-it) · [7. Viva questions](#7-questions-an-examiner-is-likely-to-ask-with-short-answers) · [8. Limitations](#8-limitations-said-honestly)

---

## 1. The project in one minute

**The problem.** An online marketplace (Olist, Brazil) promises a delivery date when you check out. About **7 in 100 orders arrive late**, and late customers leave a 1★ or 2★ review **65%** of the time, against 9.5% when the order is on time. If the marketplace could spot risky orders *at checkout*, it could act early: expedite shipping, give an honest date, or offer a voucher.

**What we built.**
* A clean, one-row-per-order dataset from 9 messy tables (99,441 orders), where **every feature only uses information available at checkout**.
* Four classical models (logistic regression, decision tree, random forest, LightGBM), tuned and judged the way a real deployment would work: always predicting the future from the past.
* A money layer that decides, order by order, whether acting is worth the cost.
* A second model (Task B) that predicts which delivered customers will leave a bad review.
* A script (`predict.py`) that scores new orders.

**What we found.**

| | Result |
|---|---|
| Best model (chosen on validation) | Regularised logistic regression with 42 features |
| On the unseen test months (3.8% late) | Finds late orders at **2.2× the random rate**, catches 37% of them while flagging 17% |
| Surprise | A tiny 5-feature model did **even better** on the test months (0.110 vs 0.083 PR-AUC): when the world changes, simple models travel better |
| Money | Expediting pays in high-lateness periods (+470 BRL per 1,000 orders) but not at today's costs in a calm month (−293). The model loses 40× less than "expedite everything", and monthly retraining helps |
| Hidden insight | For 3 in 4 late orders, the review request arrives **before the parcel**, so only a checkout-time model (Task A) can prevent those bad reviews |
| Task B | Calling the 20 riskiest customers each day reaches **2.2× more** future bad reviewers than random calls |

---

## 2. The story, stage by stage

Each notebook follows the same pattern: **question → evidence → finding → decision.** You can read the markdown cells alone and follow the argument.

### Stages 1–2: Look at the data before trusting it (`01_data_audit`)
* **Did:** profiled all 9 tables, checked every key and join, and looked for impossible timestamps.
* **Found:** dates are stored day-first (they would silently mis-parse); a naive geolocation join would create 18 million rows; the last weeks of data are *censored* (slow orders hadn't arrived yet); the late rate swings wildly by month.
* **Decided:** strict date parsing; a compact zip-code map; the population ends at Aug 2018; split by time.

### Stage 3: Build features without cheating (`02_data_preparation`)
* **Did:** joined everything to one row per order (98,666 × 74) and defined `known_at`, the moment an outcome becomes known. Built seller, route and state history "as of" each purchase.
* **Found:** a seller late rate computed on all the data looks great (ROC 0.709) but is fake; the honest as-of version scores 0.552.
* **Decided:** every history feature counts only outcomes known before the purchase, and tests prove it.

### Stage 4: Ask the data questions (`03_eda`)
Ten questions, each ending in a decision. The highlights:
* Distance and destination matter: the late rate runs from 2.4% (RO) to 24% (AL).
* Promises are padded: median 25 days promised, 11 delivered.
* Late orders get 7× more bad reviews, and the jump happens exactly on the promised day.
* The same calendar month behaves differently across years, so **`purchase_month` is a memorisation trap**.

### Stages 5–7: Preprocess, handle imbalance, choose features (`04_preprocessing_and_imbalance`)
* **Preprocessing:** log transforms, winsorising and scaling for linear models; raw values for trees. Everything is fitted on training rows only.
* **Imbalance:** compared 5 strategies. **None ranked better than a plain fit**, and SMOTE/weights wrecked the probabilities, so we train on the real ratio.
* **Features:** group ablations dropped `purchase_month` and the product group; a stationarity audit dropped 5 "clock" features. **57 → 42 features.**

### Stage 8: Set the bar, and see the boundaries (`05_baselines_and_decision_boundaries`)
* Three baselines: majority (88% accurate, useless), a business rule, and a 5-feature logistic regression.
* A deliberately leaky model shows +87% fake improvement.
* **Pictures of decision boundaries:** a straight line, a plane and a 96-D hyperplane for logistic regression; rectangles for trees.

### Stages 9–10: Train and tune four model families (`06_models_and_tuning`)
* Randomised search for logistic regression, tree and forest; Bayesian search (Optuna) for LightGBM. Every trial is scored on walk-forward months.
* Heavy regularisation made logistic regression the winner on validation (0.221). The forest and LightGBM overfit (training scores 0.49 and 0.37).

### Stage 11: The one big exam (`07_evaluation_and_explainability`)
* **Test results:** see section 1, including the honest surprise that the small baseline wins.
* **Explanations:** model weights, tree rules, permutation importance, SHAP for whole datasets and single orders.
* **Error analysis:** the South is the hardest region; the most confident mistakes are "boring" orders that went wrong after checkout.

### Stage 12: Turn scores into decisions (`08_decision_analysis`)
* A cost rule per order, profit curves, sensitivity to every assumption, a break-even map, and a drift experiment (static vs monthly recalibration vs monthly retraining).
* Built the deployable model and `predict.py`.

### Task B: Who will leave a bad review? (`09_task_b_review_risk`)
* Found that late customers usually get the review request *before* the parcel.
* The model reaches 2.2× the base rate; multi-item orders are the biggest risk.

---

## 3. The ideas you need, in plain words

<details open>
<summary><b>Classification and probability</b></summary>

The model reads facts about an order (distance, promised days, the seller's track record, and so on) and outputs a number between 0 and 1: the **probability that the order will be late**. "Flag the order" means that probability is above some cut-off (the *threshold*).
</details>

<details>
<summary><b>Class imbalance: why 93% accuracy means nothing</b></summary>

Only about 7 in 100 orders are late. A "model" that always says *on time* is right 93% of the time and catches **zero** late orders. That's why accuracy is never the main metric here. The project uses:
* **Recall:** of all the late orders, how many did we catch?
* **Precision:** of all the orders we flagged, how many were really late?
* **PR-AUC (average precision):** the precision–recall trade-off summarised over every threshold. A random guess scores the base rate (about 0.04–0.12 depending on the month); higher is better.
</details>

<details>
<summary><b>Data leakage: the trap that makes bad models look great</b></summary>

Leakage means the model sees information it would *not* have at prediction time. Example: "carrier handoff days" is only known *after* checkout. Adding two such columns made the validation PR-AUC jump from 0.188 to 0.352 (+87%), and all of that gain was fake (notebook 05). The project prevents leakage with:
* a **feature registry** listing every allowed input and *when it is known* (`features.py`);
* a **forbidden list** (targets, delivery dates, status, and so on);
* **as-of features:** a seller's late rate is computed only from orders whose outcome was already known before this purchase;
* **tests** that change all future outcomes and check that past features don't move.
</details>

<details>
<summary><b>"As-of" (point-in-time) features and <code>known_at</code></b></summary>

When does the marketplace learn that an order was late? Not at delivery, but **the day after the promised date** passes without delivery. An on-time order is known when it's delivered. So:

`known_at = min(delivered time, promised date + 1 day)`

A history feature for an order bought at time *t* only counts orders with `known_at < t`.
</details>

<details>
<summary><b>Why split by time, and what "walk-forward" means</b></summary>

A random split would let the model train on March and be tested on February. That is cheating, because the model would know the future. Instead:
* **train** = Jan 2017 – Feb 2018, **validation** = Mar–Apr 2018, **test** = May–Jul 2018;
* the **test set is used exactly once**, at the end;
* **walk-forward CV:** to tune inside the training period, train on everything up to October and test on November, then train up to November and test on December, and so on. It always predicts the future from the past;
* **maturity purge:** if an order's promised date hadn't passed by the training cutoff, its label wasn't known yet, so it's removed from training.
</details>

<details>
<summary><b>Decision boundary and hyperplane</b></summary>

A classifier splits the feature space into "flag" and "don't flag" regions. The border is the **decision boundary**.
* **Logistic regression** computes a weighted sum `w·x + b`. Its boundary is where that sum equals a constant: a straight line in 2-D, a flat plane in 3-D, and a **hyperplane** in many dimensions (here 96).
* **Decision trees** ask yes/no questions on one feature at a time, so their boundaries are **rectangles**.
* **Random forests and boosting** combine many trees, so their boundaries are **stepped and bendy**.

See `reports/figures/05_decision_boundaries_2d.png`, `05_hyperplane_3d.png` and `05_hyperplane_full_space.png`.
</details>

<details>
<summary><b>Hyperparameter tuning</b></summary>

Settings that are not learned from data (tree depth, learning rate, regularisation strength) are *hyperparameters*. The project tries many combinations:
* **randomised search** (logistic regression, tree, forest);
* **Optuna's Bayesian search** (LightGBM), which learns from earlier trials which region to try next.

Every trial is scored with walk-forward CV, never on validation or test.
</details>

<details>
<summary><b>Calibration</b></summary>

If the model says "30% risk" for 100 orders, about 30 of them should really be late. If not, the probabilities are *mis-calibrated*. Calibration (Platt/sigmoid or isotonic) fixes this on held-out data. It matters because the business decision compares the probability with money.
</details>

<details>
<summary><b>The cost rule: when is expediting worth it?</b></summary>

Expediting order *i* costs `E = 2 BRL + 0.5 × freight`. A late order causes damage `L = 0.5 × (price + freight)`, and expediting saves a late order half the time (`e = 0.5`). Acting is worth it when the expected saving beats the cost:

`p × e × L > E` ⇔ `p > E / (e × L)`

So every order gets its **own threshold**: a valuable parcel with cheap freight is worth protecting at a lower risk. All three numbers are assumptions, which is why notebook 08 varies each by ±50%.
</details>

<details>
<summary><b>SHAP explanations</b></summary>

SHAP splits a single prediction into "pushes" from each feature. For example: "this order's risk is high mainly because it goes to Alagoas, a state with a high late rate, from a seller with a poor track record, 1,900 km away." Notebook 07 shows this for a correct alert, a missed late order and a false alarm.
</details>

<details>
<summary><b>Drift</b></summary>

The late rate changes over time (3% in some months, 14% in others). A model trained once slowly goes stale. Notebook 08 compares *train once* with *recalibrate monthly* and *retrain monthly*.
</details>


---

## 4. How to run it

You need Python 3.11+ and the 9 Olist CSVs in `Dataset/` (see the README for the timestamp-format note).

```bash
pip install -r requirements.txt
```

Rebuild **everything**: data checks, all 9 notebooks (figures, tables, models) and the tests. This takes about 30–45 minutes, most of it hyperparameter tuning.

```bash
python run_all.py
```

Resume from a notebook, or run only some of them:

```bash
python run_all.py --from 07
```

Score orders that are still in transit with the deployed model:

```bash
python predict.py
```

Run the tests: 35 in total, of which 30 run in about 4 seconds without the full data build.

```bash
python -m pytest
```

## 5. Where everything lives

| Path | What it is | Open it when you want to… |
|---|---|---|
| `Olist Delivery Promise Risk ML Topic Proposal.md` | The original plan (12 stages) | see what was promised |
| `notebooks/01_data_audit.ipynb` | Stages 1–2: tables, keys, data quality, drift, censoring | understand the raw data |
| `notebooks/02_data_preparation.ipynb` | Stage 3: one row per order, labels, as-of history, split | see how features are built without leakage |
| `notebooks/03_eda.ipynb` | Stage 4: 10 questions → findings → decisions | explain *why* each modelling choice was made |
| `notebooks/04_preprocessing_and_imbalance.ipynb` | Stages 5–7: pipelines, imbalance strategies, feature selection, stationarity audit | defend the feature set and the "no SMOTE" decision |
| `notebooks/05_baselines_and_decision_boundaries.ipynb` | Stage 8: baselines, leaky model, decision boundaries and hyperplanes | show the hyperplane pictures |
| `notebooks/06_models_and_tuning.ipynb` | Stages 9–10: four model families, randomised search + Optuna | show tuning and overfitting curves |
| `notebooks/07_evaluation_and_explainability.ipynb` | Stage 11: test results, curves, calibration, bootstrap, SHAP, error slices | present the final scores and explanations |
| `notebooks/08_decision_analysis.ipynb` | Stage 12: cost rule, profit curves, sensitivity, drift, deployable model | present the business value |
| `notebooks/09_task_b_review_risk.ipynb` | Task B: bad-review risk at delivery | present the secondary task |
| `reports/*.md` | One short written report per stage | read the findings without running code |
| `reports/figures/` | Every figure, prefixed with its notebook number | build slides |
| `reports/results/` | Every result table as CSV/JSON | quote exact numbers |
| `src/olist/` | The reusable, tested package | read the actual logic |
| `predict.py` | Scores new orders | demo the model |
| `run_all.py` | One-command rebuild | prove reproducibility |
| `tests/` | 35 tests | prove the leakage controls work |
| `SOURCES.md` | Data licence, libraries, AI-use log | cite sources |


---

## 6. What went wrong, and how we fixed it

Real projects hit problems. Explaining them well is often worth more marks than pretending they didn't happen. Every one of these is kept visible in the notebooks.

| Problem | How we noticed | Fix |
|---|---|---|
| **Dates mis-parsed silently.** `02-10-2017` could mean 2 Oct or 10 Feb | Audit of raw formats (notebook 01) | Explicit format; the loader *fails loudly* on any mismatch |
| **"Delivered before purchase" would under-count late orders**, because late outcomes resolve later | Reasoning about *when* outcomes become known | `known_at = min(delivered, promised + 1 day)` |
| **A purge rule that selects on the label.** Keeping only "resolved" orders kept only early deliveries | The purged rows were 26.6% late | Purge on a rule known at checkout (promise + 1 day ≤ cutoff) |
| **`purchase_month` looked helpful in CV** | Pooled cross-month scores collapsed (0.155 → 0.123) | Dropped. It memorised one-off months |
| **LightGBM tuning produced a 10-tree model.** Early stopping watched the month being scored | Validation was worse than the untuned model | Early stopping on a separate inner holdout |
| **Then a 2-tree model.** Upweighting late orders broke the log-loss stopping signal | Best trial had `n_estimators = 2` | No class weighting (as Stage 6 had decided) |
| **"Clock" features.** Market volume counts rise with time, so models learned "busy period = late" | A first look at the test set; then label-free checks (ρ = 0.95 with date, adversarial AUC 1.0) | Removed 5 clock features; **disclosed** that the test set was looked at twice |
| **The simple baseline still won on test** | Final evaluation | Not "fixed": that would mean tuning on the test set. Reported, diagnosed (parsimony under drift), and turned into a lesson |


---

## 7. Questions an examiner is likely to ask, with short answers

**Q: Why not use accuracy?**
Only about 7% of orders are late, so predicting "on time" for everything scores about 93% accuracy (88% on validation) and catches nothing. PR-AUC, recall and precision on the late class measure what matters.

**Q: Why did you split by time instead of randomly?**
The late rate drifts from 3% to 14% between months. A random split would let the model learn from the future and hide that drift. Real deployment always predicts tomorrow from yesterday.

**Q: How do you know there is no leakage?**
Every feature is listed in a registry with *when it is known*, and the forbidden columns are listed separately. A test asserts the two never overlap. History features use only outcomes known before the purchase (`known_at < t`). A test randomises every outcome after 2018-01-01 and checks that earlier features don't change at all. Notebook 05 also shows what leakage would have done: +87% fake PR-AUC.

**Q: Why didn't SMOTE help?**
On these folds it did not improve ranking (PR-AUC equal or lower), and it badly distorted the probabilities (Brier score 0.092 → 0.42 for logistic regression), because the model is trained on a 50/50 world. The project needs honest probabilities for the cost decision, so it trains on the real class ratio and handles imbalance through the metric, the threshold and the cost rule.

**Q: Why did you drop `purchase_month` if the ablation said it helped?**
Each CV fold is one month, so within a fold the month is the same for every order and can't change the ranking; its "help" came from acting as a time fixed effect during training. When the folds are pooled into one ranking across months, the month makes things much worse (logistic regression pooled PR-AUC 0.155 → 0.123). It memorises one-off months: February was 3% late in 2017 but 14% in 2018.

**Q: What is a hyperplane in your model?**
Logistic regression flags an order when `w·x + b` exceeds a constant. The set of points where it exactly equals that constant is a flat surface: a line in 2-D, a plane in 3-D, and a hyperplane in the 96-dimensional one-hot feature space. Notebook 05 draws all three views.

**Q: Why is test PR-AUC lower than validation PR-AUC?**
PR-AUC depends on the base rate. Validation has 11.8% late orders and test has only 3.8%, so the same model gets a lower PR-AUC on test. Compare each number with its base rate (the *lift*), not across periods.

**Q: How did you choose the threshold?**
Two ways. For the classification metrics, the F1-optimal threshold was chosen on validation and applied unchanged to test. For the business decision, no threshold is tuned at all: each order's threshold is `E / (e·L)`, derived from its own cost and value.

**Q: What if your cost assumptions are wrong?**
That's why notebook 08 has a tornado chart: each assumption is moved ±50% and the policy is re-derived. The report states in which scenarios the model still beats "do nothing" and "expedite everything".

**Q: Is this causal? Does expediting really prevent lateness?**
No. The data is observational. The model predicts risk; it cannot prove that expediting helps. The effectiveness `e` is an assumption and would need an A/B test.

**Q: How would this run in production?**
`predict.py` rebuilds the features from the raw tables using exactly the training code, loads the calibrated model, and outputs a probability, a per-order threshold, an action and the top risk factors for every order in transit. The drift experiment suggests retraining or recalibrating monthly.

**Q: Your best model lost to a 5-feature baseline on the test set. Isn't that a failure?**
It's the most useful finding of the project. The tuned models won on validation (0.22 vs 0.14) but lost on test (0.08 vs 0.11). A 4-feature model without the month matches the baseline, so the cause is parsimony: under a regime change (11.8% late → 3.8%), physical features (distance, promise, seller reliability) transfer, while detailed geography learned in a high-lateness period does not. We reported it rather than tuning on the test set. The fix for production is to select models across several time periods and retrain monthly.

**Q: You said the test set is used once, but you looked at it twice. Why is that acceptable?**
It was disclosed, not hidden. The first look exposed "clock" features. The fix used label-free evidence (correlation with the date, adversarial validation) and a stated principle (features must describe the order, not the calendar), not test scores. We then stopped iterating. Both sets of numbers are in notebook 07, and the final ones are described as slightly optimistic.

**Q: What is adversarial validation?**
Train a classifier to tell training rows from validation rows. If it can (AUC near 1), some features differ systematically between periods. Its most important features show *which* ones. Here it pointed straight at the market-volume counters.

**Q: What does the model recommend the business should actually do?**
Retrain monthly and recalibrate. Expedite only high-value orders, and only when lateness is running high. Use the model freely for cheap actions such as honest ETAs. Run an A/B test to measure whether expediting really helps.

**Q: What is the most surprising thing in the data?**
For 76% of late orders the review request arrives *before* the parcel, and three in four of those customers leave 1–2★. So post-delivery apologies come too late, and prevention at checkout is the only lever.


---

## 8. Limitations, said honestly

* **One marketplace, one country, 2016–2018.** The results may not transfer to other markets or years.
* **The costs are assumptions.** Expedite cost, damage and effectiveness are educated guesses; the sensitivity analysis shows how much the conclusion depends on them.
* **No causal claim.** The data is observational. Nobody expedited orders, so the benefit of acting is assumed, not measured. An A/B test would be needed.
* **Strong drift.** Late rates swing between 1% and 14% by month. Validation (11.8% late) and test (3.8%) look very different, so thresholds and calibration do not transfer cleanly. This is measured, not hidden.
* **The test set was looked at twice.** A first evaluation revealed clock-like features. They were removed using label-free checks and the test was re-run. Both sets of numbers are reported.
* **Licence.** CC BY-NC-SA 4.0: fine for coursework, not for commercial use.

**Future work:** uplift modelling with experimental data, Portuguese review text as an NLP feature, carrier information, and testing on other (for example Indian) e-commerce logistics data.


---

*Every number in this guide comes from an executed notebook. If you change anything, re-run `python run_all.py` and the notebooks, reports and figures regenerate.*
