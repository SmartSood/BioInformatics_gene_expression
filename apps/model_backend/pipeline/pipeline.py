from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Tuple, List
import os

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
import warnings
import sys
from io import StringIO
import traceback
import logging
import json

logger = logging.getLogger(__name__)

# Configure MLflow tracking URI to use absolute path
# This ensures runs are always stored in the same location regardless of where the worker runs
_MLFLOW_DIR = Path(__file__).resolve().parent.parent / "mlruns"
_MLFLOW_DIR.mkdir(parents=True, exist_ok=True)
_MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", str(_MLFLOW_DIR))
mlflow.set_tracking_uri(_MLFLOW_TRACKING_URI)

# Ensure default experiment exists
# MLflow uses experiment ID 0 as the default experiment
# Create the directory structure if it doesn't exist
_experiment_dir = _MLFLOW_DIR / "0"
_experiment_dir.mkdir(parents=True, exist_ok=True)

# Create meta.yaml for experiment 0 if it doesn't exist
_meta_file = _experiment_dir / "meta.yaml"
if not _meta_file.exists():
    _meta_file.write_text("""artifact_location: {artifact_location}
experiment_id: '0'
lifecycle_stage: active
name: Default
""".format(artifact_location=str(_MLFLOW_DIR / "0")))

# Ensure we're using the default experiment
try:
    mlflow.set_experiment("Default")
except Exception:
    # If Default doesn't exist, create it
    try:
        mlflow.create_experiment("Default")
        mlflow.set_experiment("Default")
    except Exception:
        # Fallback: use experiment ID 0 directly
        pass

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
        # Use more lenient threshold if not specified - use "median" instead of "mean" (default)
        # This is less aggressive and helps avoid removing all features
        threshold = cfg.get("importance_threshold")
        if threshold is None:
            # Use "median" which is less aggressive than "mean" (the default)
            # Or use a small negative value to be more lenient
            threshold = "median"  # This selects features with importance >= median
        
        if problem_type == "classification":
            from sklearn.linear_model import LogisticRegression
            est = LogisticRegression(penalty="l1", solver="liblinear", C=1.0/cfg.get("alpha", 0.001), max_iter=1000)
        else:
            from sklearn.linear_model import Lasso
            est = Lasso(alpha=cfg.get("alpha", 0.001), max_iter=1000)
        return SelectFromModel(est, threshold=threshold)
    if method == "random_forest_importance":
        # Use more lenient threshold if not specified
        threshold = cfg.get("importance_threshold")
        if threshold is None:
            threshold = "median"  # Less aggressive than "mean"
        
        if problem_type == "classification":
            from sklearn.ensemble import RandomForestClassifier as RF
            est = RF(n_estimators=200, random_state=42)
        else:
            from sklearn.ensemble import RandomForestRegressor as RF
            est = RF(n_estimators=200, random_state=42)
        return SelectFromModel(est, threshold=threshold)
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

    # Capture warnings
    warnings_capture = []
    warnings_log = StringIO()
    
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        with mlflow.start_run() as run:
            # Log high-level params
            mlflow.log_params({
                "model": config["model"],
                "problem_type": problem_type,
                "cv_folds": cv_folds,
                **config.get("hyperparams", {})
            })
            
            # Log feature selection info if used
            if selector is not None:
                fs_cfg = prep.get("feature_selection", {})
                fs_method = fs_cfg.get("method", "none")
                mlflow.log_param("feature_selection.method", fs_method)
                
                # Warn if dataset has very few features and using aggressive feature selection
                if X_tr.shape[1] < 5 and fs_method in ["lasso", "variance_threshold"]:
                    warning_msg = (
                        f"Warning: Dataset has only {X_tr.shape[1]} features but using {fs_method} feature selection. "
                        f"This may result in all features being removed. Consider using a different method or disabling feature selection."
                    )
                    warnings_capture.append(warning_msg)
                    logger.warning(warning_msg)
                
                if fs_method != "none":
                    if fs_cfg.get("k_features"):
                        mlflow.log_param("feature_selection.k_features", str(fs_cfg.get("k_features")))
                    if fs_cfg.get("variance_threshold") is not None:
                        mlflow.log_param("feature_selection.variance_threshold", str(fs_cfg.get("variance_threshold")))
                    if fs_cfg.get("alpha") is not None:
                        mlflow.log_param("feature_selection.alpha", str(fs_cfg.get("alpha")))
                    if fs_cfg.get("importance_threshold") is not None:
                        mlflow.log_param("feature_selection.importance_threshold", str(fs_cfg.get("importance_threshold")))

            # Early validation: Check if preprocessing would result in empty features
            # This gives better error messages before attempting CV
            try:
                # Fit the preprocessing steps to see output shape
                if "prep" in pipe.named_steps:
                    prep_step = pipe.named_steps["prep"]
                    X_tr_prep = prep_step.fit_transform(X_tr)
                    
                    # Check feature selection if used
                    if "feature_select" in pipe.named_steps:
                        fs_step = pipe.named_steps["feature_select"]
                        fs_step.fit(X_tr_prep, y_tr)
                        X_tr_final = fs_step.transform(X_tr_prep)
                        
                        if X_tr_final.shape[1] == 0:
                            error_msg = (
                                f"Feature selection resulted in 0 features. "
                                f"Original features after preprocessing: {X_tr_prep.shape[1]}, "
                                f"Method: {fs_cfg.get('method', 'unknown')}. "
                                f"This may be due to too strict feature selection criteria."
                            )
                            warnings_capture.append(error_msg)
                            mlflow.log_param("error", error_msg[:500])
                            mlflow.log_metric("n_features_original", float(X_tr_prep.shape[1]))
                            mlflow.log_metric("n_features_selected", 0.0)
                            raise ValueError(error_msg)
                        else:
                            # Log successful feature selection
                            mlflow.log_metric("n_features_original", float(X_tr_prep.shape[1]))
                            if hasattr(fs_step, 'get_support'):
                                n_selected = int(np.sum(fs_step.get_support()))
                                mlflow.log_metric("n_features_selected", float(n_selected))
                    else:
                        mlflow.log_metric("n_features_original", float(X_tr_prep.shape[1]))
            except ValueError:
                # Re-raise validation errors
                raise
            except Exception as e:
                # Log validation errors but continue - CV will catch them
                validation_error = f"Pre-validation warning: {str(e)}"
                warnings_capture.append(validation_error)
                logger.warning(validation_error)

            # Cross-validation on training split
            scoring = "accuracy" if problem_type == "classification" else "r2"
            
            try:
                cv_scores = cross_val_score(pipe, X_tr, y_tr, cv=cv_folds, scoring=scoring, error_score='raise')
                cv_mean = float(np.mean(cv_scores))
                cv_std = float(np.std(cv_scores))
                
                # Check for NaN scores (indicates failures)
                nan_count = np.sum(np.isnan(cv_scores))
                if nan_count > 0:
                    mlflow.log_metric("cv_nan_count", float(nan_count))
                    raise ValueError(f"{nan_count} out of {cv_folds} CV folds failed. This may indicate data quality issues or feature selection removed all features.")
                
                mlflow.log_metric("cv_mean", cv_mean)
                mlflow.log_metric("cv_std", cv_std)
            except ValueError as e:
                # Check if it's the "0 features" error - provide specific diagnostics
                error_str = str(e)
                if "0 feature" in error_str or "minimum of 1 is required" in error_str:
                    # This is a feature selection issue
                    diagnostics_msg = (
                        f"Feature selection removed all features during CV. "
                        f"Method: {fs_cfg.get('method', 'unknown')}, "
                        f"Original features: {X_tr.shape[1]}. "
                        f"This suggests the feature selection criteria are too strict for this dataset. "
                        f"Try: (1) Using a different feature selection method, "
                        f"(2) Relaxing the selection parameters (e.g., lower variance_threshold, lower alpha), "
                        f"or (3) Disabling feature selection."
                    )
                    # Log detailed diagnostics to MLflow
                    mlflow.log_param("error_type", "feature_selection_removed_all")
                    mlflow.log_param("error_details", diagnostics_msg[:500])
                    mlflow.log_metric("n_features_original", float(X_tr.shape[1]))
                    mlflow.log_metric("n_features_selected", 0.0)
                    if selector is not None:
                        fs_cfg = prep.get("feature_selection", {})
                        mlflow.log_param("feature_selection.method", fs_cfg.get("method", "none"))
                        if fs_cfg.get("variance_threshold") is not None:
                            mlflow.log_param("feature_selection.variance_threshold_used", str(fs_cfg.get("variance_threshold")))
                        if fs_cfg.get("alpha") is not None:
                            mlflow.log_param("feature_selection.alpha_used", str(fs_cfg.get("alpha")))
                    
                    error_msg = f"{diagnostics_msg}\n\nOriginal error: {error_str}"
                else:
                    error_msg = f"Cross-validation failed: {error_str}\n{traceback.format_exc()}"
                
                warnings_capture.append(error_msg)
                mlflow.log_param("cv_error", error_msg[:500])  # Log truncated error
                
                # Log warnings before failing
                if warnings_capture:
                    warnings_str = "\n".join(warnings_capture)
                    try:
                        mlflow.log_text(warnings_str, artifact_file="warnings.txt")
                        mlflow.log_param("warnings_count", str(len(warnings_capture)))
                    except Exception:
                        pass
                
                raise ValueError(error_msg) from e
            except Exception as e:
                # Log other errors
                error_msg = f"Cross-validation failed: {str(e)}\n{traceback.format_exc()}"
                warnings_capture.append(error_msg)
                mlflow.log_param("cv_error", str(e)[:500])  # Log truncated error
                
                # Log warnings before failing
                if warnings_capture:
                    warnings_str = "\n".join(warnings_capture)
                    try:
                        mlflow.log_text(warnings_str, artifact_file="warnings.txt")
                        mlflow.log_param("warnings_count", str(len(warnings_capture)))
                    except Exception:
                        pass
                
                raise ValueError(error_msg) from e

            # Collect warnings
            for warning in w:
                warning_msg = f"{warning.category.__name__}: {str(warning.message)}"
                warnings_capture.append(warning_msg)
                warnings_log.write(warning_msg + "\n")

            # Fit on full training split, evaluate on test
            feature_selection_info = {}
            try:
                n_features_before = X_tr.shape[1]
                pipe.fit(X_tr, y_tr)
                
                # Log feature selection results if used (after fitting)
                if selector is not None:
                    # Try to get number of features selected and which features
                    try:
                        # After fitting, check the transformed shape
                        if "feature_select" in pipe.named_steps:
                            # Get the feature selector from the pipeline
                            fs_step = pipe.named_steps["feature_select"]
                            
                            # Get feature names after preprocessing
                            prep_step = pipe.named_steps["prep"]
                            X_tr_prep = prep_step.transform(X_tr)
                            
                            # Try to get feature names - this depends on the transformer output
                            feature_names_after_prep = None
                            try:
                                if hasattr(X_tr_prep, 'columns'):
                                    feature_names_after_prep = X_tr_prep.columns.tolist()
                                elif hasattr(prep_step, 'get_feature_names_out'):
                                    feature_names_after_prep = prep_step.get_feature_names_out().tolist()
                                elif hasattr(X_tr_prep, 'shape'):
                                    # Fallback: use indices
                                    feature_names_after_prep = [f"feature_{i}" for i in range(X_tr_prep.shape[1])]
                            except Exception:
                                feature_names_after_prep = [f"feature_{i}" for i in range(X_tr_prep.shape[1])]
                            
                            if hasattr(fs_step, 'get_support'):
                                # Get support after fit
                                support = fs_step.get_support()
                                n_features_selected = int(np.sum(support))
                                
                                # Get selected feature names/indices
                                selected_indices = np.where(support)[0].tolist()
                                if feature_names_after_prep:
                                    selected_feature_names = [feature_names_after_prep[i] for i in selected_indices]
                                else:
                                    selected_feature_names = selected_indices
                                
                                # Store in feature_selection_info
                                feature_selection_info = {
                                    "n_features_original": int(n_features_before),
                                    "n_features_selected": n_features_selected,
                                    "selected_feature_indices": selected_indices,
                                    "selected_feature_names": selected_feature_names[:100],  # Limit to first 100 to avoid huge JSON
                                }
                                
                                mlflow.log_metric("n_features_selected", float(n_features_selected))
                                mlflow.log_metric("n_features_original", float(n_features_before))
                                mlflow.log_param("n_features_selected", str(n_features_selected))
                                
                                # Log selected features as JSON string (MLflow params have size limits)
                                try:
                                    features_json = json.dumps(selected_feature_names[:50])  # First 50 features
                                    if len(selected_feature_names) > 50:
                                        features_json += f" ... and {len(selected_feature_names) - 50} more"
                                    mlflow.log_param("selected_features_sample", features_json[:500])  # Truncate to 500 chars
                                except Exception:
                                    pass
                                
                            elif hasattr(fs_step, 'n_features_'):
                                n_selected = fs_step.n_features_
                                feature_selection_info = {
                                    "n_features_original": int(n_features_before),
                                    "n_features_selected": int(n_selected) if n_selected is not None else 0,
                                }
                                mlflow.log_metric("n_features_selected", float(n_selected) if n_selected is not None else 0.0)
                                mlflow.log_metric("n_features_original", float(n_features_before))
                                mlflow.log_param("n_features_selected", str(n_selected) if n_selected is not None else "0")
                            elif hasattr(fs_step, 'n_features_to_select'):
                                n_selected = fs_step.n_features_to_select
                                feature_selection_info = {
                                    "n_features_original": int(n_features_before),
                                    "n_features_selected": int(n_selected) if n_selected is not None else 0,
                                }
                                mlflow.log_metric("n_features_selected", float(n_selected) if n_selected is not None else 0.0)
                                mlflow.log_metric("n_features_original", float(n_features_before))
                                mlflow.log_param("n_features_selected", str(n_selected) if n_selected is not None else "0")
                    except Exception as e:
                        # If we can't get feature selection info, log a warning but continue
                        warning_msg = f"Could not extract feature selection metrics: {str(e)}"
                        warnings_capture.append(warning_msg)
                        logger.warning(warning_msg)
                        # Still store basic counts
                        feature_selection_info = {
                            "n_features_original": int(n_features_before),
                            "n_features_selected": None,
                            "extraction_error": str(e)[:200]
                        }
                else:
                    # No feature selection - store original count
                    feature_selection_info = {
                        "n_features_original": int(n_features_before),
                        "n_features_selected": int(n_features_before),
                    }
                    mlflow.log_metric("n_features_original", float(n_features_before))
                    mlflow.log_metric("n_features_selected", float(n_features_before))
                
            except Exception as e:
                error_msg = f"Model fitting failed: {str(e)}\n{traceback.format_exc()}"
                warnings_capture.append(error_msg)
                mlflow.log_param("fit_error", str(e)[:500])
                raise ValueError(error_msg) from e
                
            if problem_type == "classification":
                try:
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
                except Exception as e:
                    error_msg = f"Prediction failed: {str(e)}\n{traceback.format_exc()}"
                    warnings_capture.append(error_msg)
                    raise ValueError(error_msg) from e
            else:
                try:
                    preds = pipe.predict(X_te)
                    mse = mean_squared_error(y_te, preds)
                    metrics = {"r2": float(r2_score(y_te, preds)), "rmse": float(mse ** 0.5)}
                except Exception as e:
                    error_msg = f"Prediction failed: {str(e)}\n{traceback.format_exc()}"
                    warnings_capture.append(error_msg)
                    raise ValueError(error_msg) from e

            for k, v in metrics.items():
                mlflow.log_metric(k, v)

            # Log warnings if any
            if warnings_capture:
                warnings_str = "\n".join(warnings_capture)
                mlflow.log_text(warnings_str, artifact_file="warnings.txt")
                mlflow.log_param("warnings_count", str(len(warnings_capture)))

            # Persist
            Path(artifacts_dir).mkdir(parents=True, exist_ok=True)
            model_path = str(Path(artifacts_dir) / "model.joblib")
            dump(pipe, model_path)
            mlflow.log_artifact(model_path, artifact_path="model")

            # Build return value
            result_metrics = {**metrics, "cv_mean": cv_mean, "cv_std": cv_std}
            
            # Add feature selection info to metrics
            if feature_selection_info:
                result_metrics["feature_selection"] = feature_selection_info
                # Also add top-level counts for easy access
                result_metrics["n_features_original"] = feature_selection_info.get("n_features_original")
                result_metrics["n_features_selected"] = feature_selection_info.get("n_features_selected")
            
            # Add warnings to metrics if any
            if warnings_capture:
                result_metrics["warnings"] = warnings_capture
                result_metrics["warnings_count"] = len(warnings_capture)

            return {
                "run_id": run.info.run_id,
                "metrics": result_metrics,
                "model_path": model_path,
                "warnings": warnings_capture if warnings_capture else None,
                "feature_selection": feature_selection_info if feature_selection_info else None
            }
