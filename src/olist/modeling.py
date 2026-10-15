"""Task A modelling utilities: population loading, preprocessing pipelines,
walk-forward folds and the classical model zoo.

Every transformer lives inside a scikit-learn Pipeline, so imputation medians,
winsorising limits, scalers and target encodings are learned from training rows only
and applied unchanged to validation and test rows.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (FunctionTransformer, OneHotEncoder, OrdinalEncoder,
                                   StandardScaler, TargetEncoder)
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import validate_data

from . import config as C
from .features import FEATURES, model_features
from .split import assign_split, matured_by

TARGET = "late_day"
P = "order_purchase_timestamp"

# Non-negative, right-skewed inputs that are log1p-transformed for linear models.
LOG_FEATURES = [
    "promised_days", "promised_business_days", "n_items", "n_products", "n_sellers", "n_categories",
    "total_price", "max_item_price", "total_freight", "freight_ratio", "total_weight_g", "max_weight_g",
    "total_volume_cm3", "max_volume_cm3", "mean_description_length", "distance_km_max", "distance_km_min",
    "n_seller_states", "seller_resolved_n_min", "seller_prior_orders_min", "seller_orders_7d_max",
    "seller_orders_30d_max", "route_resolved_n_min", "market_resolved_30d", "customer_prior_orders",
]
# Low-cardinality categoricals: one-hot for linear models.
ONEHOT_FEATURES = ["customer_state", "main_seller_state", "main_payment_type",
                   "purchase_dow", "promise_dow", "purchase_month"]
# String categoricals that trees cannot split on directly.
STRING_CATS = ["customer_state", "main_seller_state", "main_category", "main_payment_type"]


# --------------------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------------------
def load_task_a(path=None) -> pd.DataFrame:
    """Primary Task A population with a `split` column and the training maturity flag."""
    df = pd.read_parquet(path or C.ORDER_DATASET_PATH)
    pop = df[df["eligible_primary"]].sort_values(P, kind="mergesort").reset_index(drop=True)
    pop["split"] = assign_split(pop[P])
    pop["train_matured"] = pop["split"].eq("train") & matured_by(pop, C.SPLITS["train"][1])
    pop["purchase_ym"] = pop[P].dt.to_period("M").astype(str)
    return pop


def task_a_sets(pop: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """train (maturity-purged), train_period (unpurged, for walk-forward CV), val, test."""
    return {
        "train": pop[pop["train_matured"]].reset_index(drop=True),
        "train_period": pop[pop["split"].eq("train")].reset_index(drop=True),
        "val": pop[pop["split"].eq("val")].reset_index(drop=True),
        "test": pop[pop["split"].eq("test")].reset_index(drop=True),
    }


def walk_forward_folds(df: pd.DataFrame, val_months=("2017-11", "2017-12", "2018-01", "2018-02")
                       ) -> list[tuple[np.ndarray, np.ndarray]]:
    """Expanding-window folds inside the training period.

    Fold k validates on every order purchased in month k. Its training rows are the orders
    purchased before that month whose promised day had passed by the month start (the same
    maturity rule as the final split), so no fold trains on a label it could not have known.
    """
    p = df[P]
    folds = []
    for m in val_months:
        start = pd.Timestamp(m)
        end = start + pd.offsets.MonthBegin(1)
        tr = (p < start) & matured_by(df, start)
        va = (p >= start) & (p < end)
        folds.append((np.flatnonzero(tr.to_numpy()), np.flatnonzero(va.to_numpy())))
    return folds


def walk_forward_predict(estimator, df: pd.DataFrame, folds, features: list[str],
                         target: str = TARGET, n_jobs: int = 1) -> list[np.ndarray]:
    """Fit a fresh clone on each fold's training rows and predict its validation month."""
    from joblib import Parallel, delayed
    from sklearn.base import clone

    X, y = df[features], df[target].astype(int)

    def run(tr, va):
        return clone(estimator).fit(X.iloc[tr], y.iloc[tr]).predict_proba(X.iloc[va])[:, 1]

    return Parallel(n_jobs=n_jobs)(delayed(run)(tr, va) for tr, va in folds)


def walk_forward_scores(estimator, df: pd.DataFrame, folds, features: list[str],
                        target: str = TARGET, n_jobs: int = 1, labels=None) -> pd.DataFrame:
    """Per-fold PR-AUC / ROC-AUC / base rate of an estimator under walk-forward CV."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    preds = walk_forward_predict(estimator, df, folds, features, target, n_jobs)
    rows = []
    for k, ((_, va), p) in enumerate(zip(folds, preds)):
        y = df[target].to_numpy()[va].astype(int)
        rows.append({"fold": labels[k] if labels else k, "pr_auc": average_precision_score(y, p),
                     "roc_auc": roc_auc_score(y, p), "base_rate": y.mean()})
    return pd.DataFrame(rows).set_index("fold")


# --------------------------------------------------------------------------------------
# preprocessing
# --------------------------------------------------------------------------------------
class Winsorizer(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    """Clip each column to quantiles learned on the training data."""

    def __init__(self, lower: float = 0.01, upper: float = 0.99):
        self.lower, self.upper = lower, upper

    def fit(self, X, y=None):
        X = validate_data(self, X, dtype=float, ensure_all_finite="allow-nan")
        self.lo_ = np.nanquantile(X, self.lower, axis=0)
        self.hi_ = np.nanquantile(X, self.upper, axis=0)
        return self

    def transform(self, X):
        X = validate_data(self, X, dtype=float, ensure_all_finite="allow-nan", reset=False)
        return np.clip(X, self.lo_, self.hi_)


def _split_kinds(features: list[str]) -> dict[str, list[str]]:
    # Features outside the Task A registry (e.g. Task B delivery facts) are numeric.
    kinds = FEATURES["kind"].reindex(features).fillna("num")
    cats = [f for f in features if kinds[f] == "cat"]
    return {
        "log": [f for f in features if f in LOG_FEATURES],
        "num": [f for f in features if kinds[f] == "num" and f not in LOG_FEATURES],
        "bin": [f for f in features if kinds[f] == "bin"],
        "onehot": [f for f in cats if f in ONEHOT_FEATURES],
        "target": [f for f in cats if f not in ONEHOT_FEATURES],
    }


def linear_preprocessor(features: list[str] | None = None) -> ColumnTransformer:
    """log1p -> median impute (+ missing flags) -> winsorise 1/99% -> standardise;
    one-hot for low-cardinality categoricals; cross-fitted target encoding for main_category."""
    k = _split_kinds(features or model_features())
    log1p = FunctionTransformer(np.log1p, feature_names_out="one-to-one")
    parts = [
        ("log", Pipeline([("log1p", log1p), ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                          ("winsor", Winsorizer()), ("scale", StandardScaler())]), k["log"]),
        ("num", Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)),
                          ("winsor", Winsorizer()), ("scale", StandardScaler())]), k["num"]),
        ("bin", "passthrough", k["bin"]),
        ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=30,
                                 sparse_output=False), k["onehot"]),
        ("target", Pipeline([("te", TargetEncoder(target_type="binary", random_state=C.SEED)),
                             ("scale", StandardScaler())]), k["target"]),
    ]
    ct = ColumnTransformer([p for p in parts if p[2]], verbose_feature_names_out=False)
    return ct.set_output(transform="pandas")


def tree_preprocessor(features: list[str] | None = None) -> ColumnTransformer:
    """Trees need no scaling and handle NaN natively. Only string categoricals are encoded:
    cross-fitted target encoding (states, category) or ordinal codes (payment type).
    Output columns keep the original feature names, one column per feature."""
    features = features or model_features()
    te = [f for f in ["customer_state", "main_seller_state", "main_category"] if f in features]
    ordinal = [f for f in ["main_payment_type"] if f in features]
    rest = [f for f in features if f not in STRING_CATS]
    parts = [
        ("rest", "passthrough", rest),
        ("target", TargetEncoder(target_type="binary", random_state=C.SEED), te),
        ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                                   encoded_missing_value=-1), ordinal),
    ]
    ct = ColumnTransformer([p for p in parts if p[2]], verbose_feature_names_out=False)
    return ct.set_output(transform="pandas")


def smotenc_sampler(extra_categorical=(), sampling_strategy: float = 1.0):
    """SMOTE-NC that decides at fit time which preprocessed columns are categorical:
    every 0/1 column (one-hot levels, flags, missing indicators) plus `extra_categorical`.
    It runs inside an imblearn Pipeline, so only training folds are ever resampled."""
    from imblearn import FunctionSampler
    from imblearn.over_sampling import SMOTENC

    def resample(X, y):
        Xd = pd.DataFrame(X)
        cat = [i for i, c in enumerate(Xd.columns)
               if c in extra_categorical or set(np.unique(Xd[c].to_numpy())) <= {0.0, 1.0}]
        sm = SMOTENC(categorical_features=cat, sampling_strategy=sampling_strategy, random_state=C.SEED)
        Xr, yr = sm.fit_resample(Xd.to_numpy(dtype=float), np.asarray(y))
        return pd.DataFrame(Xr, columns=Xd.columns), yr

    return FunctionSampler(func=resample, validate=False)


# --------------------------------------------------------------------------------------
# model zoo
# --------------------------------------------------------------------------------------
def make_model(name: str, features: list[str] | None = None, **params) -> Pipeline:
    """Preprocessing + classifier for one of: logreg, tree, forest, lgbm."""
    if name == "logreg":
        clf = LogisticRegression(solver="saga", penalty="elasticnet", l1_ratio=0.0, C=1.0,
                                 max_iter=3000, tol=1e-3, random_state=C.SEED)
        pre = linear_preprocessor(features)
    elif name == "tree":
        clf = DecisionTreeClassifier(random_state=C.SEED)
        pre = tree_preprocessor(features)
    elif name == "forest":
        clf = RandomForestClassifier(n_estimators=300, n_jobs=4, random_state=C.SEED)
        pre = tree_preprocessor(features)
    elif name == "lgbm":
        from lightgbm import LGBMClassifier
        clf = LGBMClassifier(n_estimators=400, learning_rate=0.03, random_state=C.SEED,
                             n_jobs=4, verbose=-1)
        pre = tree_preprocessor(features)
    else:
        raise KeyError(f"unknown model {name!r}")
    clf.set_params(**params)
    return Pipeline([("pre", pre), ("clf", clf)])


# Search spaces (prefix clf__ for use inside the Pipeline).
SEARCH_SPACES = {
    "logreg": {
        "clf__C": stats.loguniform(1e-3, 1e2),
        "clf__l1_ratio": stats.uniform(0, 1),
        "clf__class_weight": [None, "balanced"],
    },
    "tree": {
        "clf__max_depth": stats.randint(3, 21),
        "clf__min_samples_leaf": stats.randint(20, 1001),
        "clf__min_samples_split": stats.randint(2, 201),
        "clf__criterion": ["gini", "entropy"],
        "clf__ccp_alpha": stats.uniform(0, 0.002),
        "clf__class_weight": [None, "balanced"],
    },
    "forest": {
        "clf__n_estimators": stats.randint(200, 601),
        "clf__max_depth": [6, 10, 14, 18, 24, 30, None],
        "clf__max_features": ["sqrt", 0.3, 0.5],
        "clf__min_samples_leaf": stats.randint(5, 201),
        "clf__class_weight": [None, "balanced_subsample"],
    },
}
