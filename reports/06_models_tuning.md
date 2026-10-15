# Stages 9–10 report: classical models and hyperparameter tuning

Evidence: [`notebooks/06_models_and_tuning.ipynb`](../notebooks/06_models_and_tuning.ipynb) (executed; about 12 min). Tables: `reports/results/06_*.csv`, `06_best_params.json`. Models: `models/task_a_*.joblib`.

## Protocol

Every trial is scored with **walk-forward CV** on four training-period months (Nov 2017 → Feb 2018), using PR-AUC. The best configuration is refitted on the matured training set and compared on **validation** (Mar–Apr 2018). The test period is not touched.

| Model | Search | Trials | Best configuration |
|---|---|---:|---|
| Logistic regression (elastic net, saga) | randomised | 40 | C = 0.003, l1_ratio = 0.31, no class weight |
| Decision tree (CART) | randomised | 60 | entropy, max_depth 18, min_samples_leaf 150, ccp_alpha 0.00056 |
| Random forest | randomised | 20 | 306 trees, max_depth 18, max_features sqrt, min_samples_leaf 19 |
| LightGBM | Optuna TPE | 100 | 120 trees, learning rate 0.020, 56 leaves, max_depth 8, colsample 0.41 |

## Results

| Model | Train PR-AUC | CV PR-AUC (mean ± sd) | Validation PR-AUC | Validation ROC-AUC |
|---|---:|---:|---:|---:|
| **Logistic regression** | 0.158 | **0.183 ± 0.080** | **0.221** | 0.669 |
| LightGBM | 0.365 | 0.157 ± 0.068 | 0.217 | 0.649 |
| Random forest | 0.485 | 0.149 ± 0.059 | 0.201 | 0.635 |
| Decision tree | 0.171 | 0.138 ± 0.039 | 0.161 | 0.611 |
| Baseline: core logistic regression | 0.124 | 0.142 | 0.138 | 0.551 |
| Baseline: business rule | 0.065 | 0.107 | 0.125 | 0.526 |
| Baseline: majority | 0.056 | 0.099 | 0.118 | 0.500 |

* **Heavy regularisation wins.** The logistic regression's CV score rises from 0.152 (default settings) to 0.183. With a weak signal spread across correlated features, shrinking the weights stops the model chasing noise.
* **Overfitting is visible in the train–CV gap:** random forest 0.34, LightGBM 0.21, tree 0.03, logistic regression −0.02. The logistic regression's training score is lower than its CV score only because the training base rate (5.6%) is lower than the folds' average.
* **The decision tree plateaus from depth 3–5.** Leaf-size limits and cost-complexity pruning cut deeper trees back to the same small tree (`06_tree_depth_curve.png`).
* **No search chose class weighting**, consistent with the Stage 6 decision.
* **Selected model: logistic regression** (best validation PR-AUC). LightGBM stays as the non-linear challenger and is used for the SHAP explanations.

## Two bugs found and fixed (kept in the notebook as lessons)

1. **Early stopping on the scored month.** The first LightGBM search stopped on the very month it was scored on, using PR-AUC. On noisy months it stopped after 1–12 trees. The final model had 10 trees, an optimistic CV score (0.186) and validation 0.181, worse than the untuned model. **Fix:** stop early on an inner time-based holdout (the last 15% of each fold's training rows), using log-loss.
2. **Class weighting combined with log-loss stopping.** An intermediate search let LightGBM upweight late orders (`scale_pos_weight` 4.7). That shifts the probabilities on purpose, so the holdout log-loss worsened from the first tree and the search produced a 2-tree model. **Fix:** fix `scale_pos_weight` at 1, as Stage 6 already decided.
