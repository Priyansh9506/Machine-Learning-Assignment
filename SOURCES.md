# Sources, licences and AI-use log

## Dataset

**Brazilian E-Commerce Public Dataset by Olist**, version 2: 9 CSV tables of real, anonymised marketplace orders from 2016 to 2018.

- Kaggle: <https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce>
- Mirror: <https://github.com/olist/work-at-olist-data>
- Licence: **CC BY-NC-SA 4.0** (attribution, non-commercial, share-alike). Academic use with attribution is permitted. The trained models must not be used commercially.
- The copy used here stores `orders` and `order_items` timestamps as day-first `DD-MM-YYYY HH:MM` (see `olist.config.ORDER_TS_FORMAT`). SHA-256 checksums of the exact files are in [`reports/data_checksums.json`](reports/data_checksums.json), and `run_all.py` verifies them.

## Background references

- Olist delivery-risk dataset card, leakage list and late rate (Hugging Face): <https://huggingface.co/datasets/samuelalex37/olist-delivery-risk-data>
- Review scores of late vs on-time orders (GitHub analysis): <https://github.com/EyadMohamed07/Olist-Business-Discovery-The-Delivery-Promise-Gap>
- Machine Learning Innovative Assignment: IA Guidelines and Requirements (course document)

Figures quoted from these sources in the proposal were treated as estimates and **re-computed on the actual files** (notebook 01, section 11).

## Libraries (pinned in `requirements.txt`)

| Library | Used for |
|---|---|
| pandas, NumPy, PyArrow | loading, joins, feature engineering, parquet |
| scikit-learn | pipelines, preprocessing, logistic regression, CART, random forest, search, calibration, metrics |
| LightGBM | gradient-boosted trees |
| imbalanced-learn | undersampling and SMOTE-NC inside training folds |
| Optuna | Bayesian (TPE) hyperparameter search for LightGBM |
| SHAP | global and local explanations |
| SciPy | statistical tests (overdispersion χ², Cramér's V) |
| matplotlib, seaborn | figures |
| pytest, nbformat, nbconvert | tests and notebook execution |

XGBoost is listed in the requirements because the proposal named "XGBoost or LightGBM". LightGBM was chosen, and XGBoost is not used by any notebook.

## Methods drawn from standard practice

- Point-in-time ("as-of") features with `pandas.merge_asof`, and chronological, purged validation, both standard in time-series ML.
- Beta-binomial (Bayesian) smoothing of rates toward a prior.
- The cost-sensitive decision threshold `t* = C / (C + B)` (Elkan, 2001, *The Foundations of Cost-Sensitive Learning*).
- Platt (sigmoid) and isotonic calibration; paired bootstrap confidence intervals.
- SHAP values (Lundberg & Lee, 2017; Lundberg et al., 2020 for TreeSHAP).

## AI-use log

| Stage | AI assistance | How the output was checked |
|---|---|---|
| Proposal | Drafting support for structure and wording | Every figure quoted was re-verified on the real data in notebook 01 |
| Stages 1–3 (audit, data preparation) | Code drafting for loaders, joins and as-of history | 24 unit tests, including brute-force and future-perturbation checks; executed notebooks; row counts at every join |
| Stages 4–12 and Task B (notebooks 03–09, `modeling.py`, `evaluate.py`, `decision.py`, `baselines.py`, `scoring.py`, `predict.py`, `run_all.py`) | Claude Code (Anthropic) wrote the code, executed the notebooks, and drafted the reports, README and GUIDE from the executed outputs | Every number in the reports is copied from an executed notebook cell. New unit tests cover preprocessing (train-only fitting), the walk-forward folds (no unknown labels), the decision maths and the deployed scorer. Feature decisions that contradicted the EDA (`purchase_month`) were re-tested with an extra pooled analysis before being accepted. Two LightGBM tuning bugs were caught from validation evidence and fixed. A first look at the test set exposed "clock" features; they were removed using label-free checks, and iteration then stopped. This is disclosed in notebooks 04 and 07, which report both sets of test numbers |

The student remains responsible for the content. The suggested review is to re-run `python run_all.py`, read each notebook's findings next to its outputs, and confirm the cost assumptions in notebook 08 are acceptable.
