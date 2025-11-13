from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
from sklearn.feature_selection import VarianceThreshold, SelectFromModel, RFE, SelectKBest, chi2
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, r2_score, mean_squared_error
)
from joblib import dump
import mlflow
import importlib

# ---------------- Utility transformers -----------------
class LogTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, offset: float = 1.0):
        self.offset = offset
        self.cols_: List[str] = []
    def fit(self, X, y=None):
        self.cols_ = list(range(X.shape[1])) if not hasattr(X, 'columns') else X.select_dtypes(include=[np.number]).columns.tolist()
        return self
    def transform(self, X):
        Xc = X.copy()
        if hasattr(Xc, 'iloc'):
            num_cols = Xc.select_dtypes(include=[np.number]).columns
            Xc[num_cols] = np.log(Xc[num_cols] + self.offset)
            return Xc
        # ndarray
        return np.log(Xc + self.offset)

class QCRowFilter(BaseEstimator, TransformerMixin):
    def __init__(self, max_missing_fraction: float = None):
        self.max_missing_fraction = max_missing_fraction
        self.keep_idx_: np.ndarray | None = None
    def fit(self, X, y=None):
        if self.max_missing_fraction is None:
            self.keep_idx_ = None
            return self
        if hasattr(X, 'isna'):
            frac = X.isna().mean(axis=1).values
        else:
            frac = np.isnan(X).mean(axis=1)
        self.keep_idx_ = frac <= self.max_missing_fraction
        return self
    def transform(self, X):
        if self.keep_idx_ is None:
            return X
        return X[self.keep_idx_]

# ---------------- Model map -----------------
MODEL_MAP: Dict[str, Tuple[str, str, str]] = {
    "random_forest": ("both", "sklearn.ensemble", "RandomForestClassifier"),
    "svm": ("classification", "sklearn.svm", "SVC"),
    "neural_network": ("both", "sklearn.neural_network", "MLPClassifier"),
    "gradient_boosting": ("both", "sklearn.ensemble", "GradientBoostingClassifier"),
    "logistic_regression": ("classification", "sklearn.linear_model", "LogisticRegression"),
    "xgboost": ("both", "xgboost", "XGBClassifier"),
}

# Swap to regression counterparts when needed
REG_SWAP = {
    ("sklearn.ensemble", "RandomForestClassifier"): ("sklearn.ensemble", "RandomForestRegressor"),
    ("sklearn.neural_network", "MLPClassifier"): ("sklearn.neural_network", "MLPRegressor"),
    ("sklearn.ensemble", "GradientBoostingClassifier"): ("sklearn.ensemble", "GradientBoostingRegressor"),
    ("xgboost", "XGBClassifier"): ("xgboost", "XGBRegressor"),
}

# ---------------- Builders -----------------

def _build_scaler(method: str, feature_range):
    if method == "standard":
        return StandardScaler()
    if method == "minmax":
        return MinMaxScaler(feature_range=tuple(feature_range))
    if method == "robust":
        return RobustScaler()
    if method == "maxabs":
        return MaxAbsScaler()
    return None


def _build_feature_selector(method: str, problem_type: str, cfg: Dict[str, Any]):
    k = cfg.get("k_features")
    if method == "variance_threshold":
        return VarianceThreshold(threshold=cfg.get("variance_threshold", 0.0))
    if method == "lasso":
        # L1 model for selection
        if problem_type == "classification":
            from sklearn.linear_model import LogisticRegression
            est = LogisticRegression(penalty="l1", solver="liblinear", C=1.0/cfg.get("alpha", 0.001))
        else:
            from sklearn.linear_model import Lasso
            est = Lasso(alpha=cfg.get("alpha", 0.001))
        return SelectFromModel(est, threshold=cfg.get("importance_threshold"))
    if method == "random_forest_importance":
        if problem_type == "classification":
            from sklearn.ensemble import RandomForestClassifier as RF
            est = RF(n_estimators=200, random_state=42)
        else:
            from sklearn.ensemble import RandomForestRegressor as RF
            est = RF(n_estimators=200, random_state=42)
        return SelectFromModel(est, threshold=cfg.get("importance_threshold"))
    if method == "rfe":
        # Default base estimator depending on problem
        if problem_type == "classification":
            from sklearn.linear_model import LogisticRegression
            base = LogisticRegression(max_iter=1000)
        else:
            from sklearn.linear_model import LinearRegression
            base = LinearRegression()
        return RFE(base, n_features_to_select=k)
    if method == "chi2":
        # Requires non-negative features; apply after MinMax scaling/encoding
        return SelectKBest(score_func=chi2, k=k or 10)
    return None


def _load_estimator(problem_type: str, model_key: str, hyperparams: Dict[str, Any]):
    kind, module_name, class_name = MODEL_MAP[model_key]
    if kind == "both" and problem_type == "regression":
        module_name, class_name = REG_SWAP.get((module_name, class_name), (module_name, class_name))
    module = importlib.import_module(module_name)
    Estimator = getattr(module, class_name)
    return Estimator(**hyperparams)


def _apply_outlier_removal(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    method = cfg.get("method", "none")
    if method == "none":
        return df
    num_cols = df.select_dtypes(include=[np.number]).columns
    X = df[num_cols].copy()
    if method == "iqr":
        Q1 = X.quantile(0.25)
        Q3 = X.quantile(0.75)
        IQR = Q3 - Q1
        lo = Q1 - cfg.get("iqr_factor", 1.5) * IQR
        hi = Q3 + cfg.get("iqr_factor", 1.5) * IQR
        mask = ~((X < lo) | (X > hi)).any(axis=1)
        if cfg.get("cap_outliers", False):
            X = X.clip(lower=lo, upper=hi, axis=1)
            df[num_cols] = X
            return df
        return df.loc[mask]
    if method == "zscore":
        zthr = cfg.get("zscore_threshold", 3.0)
        z = (X - X.mean()) / X.std(ddof=0)
        mask = (np.abs(z) <= zthr).all(axis=1)
        if cfg.get("cap_outliers", False):
            X = X.clip(lower=(X.mean()-zthr*X.std()), upper=(X.mean()+zthr*X.std()), axis=1)
            df[num_cols] = X
            return df
        return df.loc[mask]
    if method == "percentile":
        pmin = cfg.get("percentile_min", 0.5) / 100.0
        pmax = cfg.get("percentile_max", 99.5) / 100.0
        lo = X.quantile(pmin)
        hi = X.quantile(pmax)
        if cfg.get("cap_outliers", False):
            X = X.clip(lower=lo, upper=hi, axis=1)
            df[num_cols] = X
            return df
        mask = ~((X < lo) | (X > hi)).any(axis=1)
        return df.loc[mask]
    return df


def train(dataset_path: str, config: Dict[str, Any], artifacts_dir: str):
    from scemma.model import TrainConfig  # for types

    df = pd.read_parquet(dataset_path) if dataset_path.endswith(".parquet") else pd.read_csv(dataset_path)

    target = config["target"]
    problem_type = config["problem_type"]
    prep = config.get("preprocessing", {})

    # --- QC filtering & optional drop rows with too many NaNs ---
    qcf = prep.get("qc_filtering", {})
    max_miss = qcf.get("max_missing_fraction", None)
    if max_miss is not None:
        frac_missing = df.isna().mean(axis=1)
        df = df.loc[frac_missing <= max_miss]

    # --- Missing value handling (drop rows vs impute later) ---
    imp_cfg = prep.get("missing_values", {})
    if imp_cfg.get("drop_rows", False):
        df = df.dropna()

    # --- Outlier removal (row-wise filters or capping) ---
    df = _apply_outlier_removal(df, prep.get("outlier_removal", {}))

    y = df[target]
    X = df.drop(columns=[target])

    # Split
    test_size = config.get("split", {}).get("test_size", 0.2)
    random_state = config.get("split", {}).get("random_state", 42)
    cv_folds = config.get("split", {}).get("cv_folds", 5)

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y if problem_type=="classification" else None)

    # Column lists
    num_cols = X_tr.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X_tr.columns if c not in num_cols]

    # Numeric pipeline
    num_steps = [
        ("imputer", SimpleImputer(strategy=imp_cfg.get("strategy_numeric", "median"))),
    ]
    # Scaling
    sc_cfg = prep.get("scaling", {})
    scaler = _build_scaler(sc_cfg.get("method", "standard"), sc_cfg.get("feature_range", [0,1]))
    if scaler:
        num_steps.append(("scaler", scaler))
    # Log transform (numeric only, pre-scaling)
    lg = prep.get("log_transform", {})
    if lg.get("enabled", False):
        # do log before scaling; apply in a separate ColumnTransformer? Simpler: add after imputer
        num_steps.insert(1, ("log", LogTransformer(offset=lg.get("offset", 1.0))))

    num_pipe = Pipeline(num_steps)

    # Categorical pipeline
    enc_cfg = prep.get("encoding", {})
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy=imp_cfg.get("strategy_categorical", "most_frequent"), fill_value=imp_cfg.get("fill_value_categorical"))),
        ("encoder", OneHotEncoder(handle_unknown="ignore", drop="first" if enc_cfg.get("drop_first", False) else None)),
    ]) if enc_cfg.get("method", "onehot") != "none" else Pipeline([
        ("imputer", SimpleImputer(strategy=imp_cfg.get("strategy_categorical", "most_frequent")))
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ])

    # Feature selection (after preprocessing)
    fs_cfg = prep.get("feature_selection", {})
    selector = _build_feature_selector(
        fs_cfg.get("method", "none"), problem_type, fs_cfg
    )

    # Estimator
    est = _load_estimator(problem_type, config["model"], config.get("hyperparams", {}))

    steps = [("prep", preprocessor)]
    if selector is not None:
        steps.append(("feature_select", selector))
    steps.append(("model", est))

    pipe = Pipeline(steps)

    with mlflow.start_run() as run:
        # Log high-level params
        mlflow.log_params({
            "model": config["model"],
            "problem_type": problem_type,
            "cv_folds": cv_folds,
            **config.get("hyperparams", {})
        })

        # Cross-validation on training split
        scoring = "accuracy" if problem_type == "classification" else "r2"
        cv_scores = cross_val_score(pipe, X_tr, y_tr, cv=cv_folds, scoring=scoring)
        mlflow.log_metric("cv_mean", float(np.mean(cv_scores)))
        mlflow.log_metric("cv_std", float(np.std(cv_scores)))

        # Fit on full training split, evaluate on test
        pipe.fit(X_tr, y_tr)
        if problem_type == "classification":
            preds = pipe.predict(X_te)
            metrics = {
                "accuracy": float(accuracy_score(y_te, preds)),
                "f1": float(f1_score(y_te, preds, average="weighted"))
            }
            proba_ok = hasattr(pipe, "predict_proba") and callable(getattr(pipe, "predict_proba"))
            if proba_ok:
                try:
                    p = pipe.predict_proba(X_te)
                    pp = p[:, 1] if p.shape[1] == 2 else p.max(axis=1)
                    metrics["roc_auc"] = float(roc_auc_score(y_te, pp))
                except Exception:
                    pass
        else:
            preds = pipe.predict(X_te)
            mse = mean_squared_error(y_te, preds)
            metrics = {"r2": float(r2_score(y_te, preds)), "rmse": float(mse ** 0.5)}

        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        # Persist
        Path(artifacts_dir).mkdir(parents=True, exist_ok=True)
        model_path = str(Path(artifacts_dir) / "model.joblib")
        dump(pipe, model_path)
        mlflow.log_artifact(model_path, artifact_path="model")

        return {
            "run_id": run.info.run_id,
            "metrics": {**metrics, "cv_mean": float(np.mean(cv_scores)), "cv_std": float(np.std(cv_scores))},
            "model_path": model_path
        }
