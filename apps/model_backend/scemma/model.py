from pydantic import BaseModel, Field
from typing import Literal, Optional, List, Dict, Any

Problem = Literal["classification", "regression"]

# ---------------- Preprocessing config blocks -----------------
class ImputationConfig(BaseModel):
    strategy_numeric: Literal["mean", "median", "most_frequent", "constant"] = "median"
    strategy_categorical: Literal["most_frequent", "constant"] = "most_frequent"
    fill_value_numeric: Optional[float] = None
    fill_value_categorical: Optional[str] = None
    drop_rows: bool = False

class OutlierRemovalConfig(BaseModel):
    method: Literal["none", "iqr", "zscore", "percentile"] = "none"
    iqr_factor: float = 1.5
    zscore_threshold: float = 3.0
    percentile_min: float = 0.5
    percentile_max: float = 99.5
    cap_outliers: bool = False

class ScalingConfig(BaseModel):
    method: Literal["none", "standard", "minmax", "robust", "maxabs"] = "standard"
    feature_range: List[float] = Field(default_factory=lambda: [0.0, 1.0])  # used for minmax
    apply_to: Literal["numeric_only", "all"] = "numeric_only"

class LogTransformConfig(BaseModel):
    enabled: bool = False
    offset: float = 1.0
    columns: Optional[List[str]] = None  # None -> all numeric

class BatchCorrectionConfig(BaseModel):
    enabled: bool = False
    method: Literal["none", "combat", "zscore", "ratio"] = "none"
    batch_column: Optional[str] = None

class QCFilteringConfig(BaseModel):
    enabled: bool = False
    max_missing_fraction: Optional[float] = 0.2  # drop rows above this
    numeric_range: Optional[Dict[str, List[float]]] = None  # {"Age": [18, 99]}

class EncodingConfig(BaseModel):
    method: Literal["onehot", "ordinal", "none"] = "onehot"
    drop_first: bool = False

# ---------------- Feature selection -----------------
FeatureSelectionMethod = Literal[
    "none",
    "variance_threshold",
    "rfe",
    "lasso",
    "random_forest_importance",
    "chi2"
]

class FeatureSelectionConfig(BaseModel):
    method: FeatureSelectionMethod = "none"
    k_features: Optional[int] = None  # used by RFE/chi2; None -> auto
    variance_threshold: float = 0.0   # used by variance_threshold
    alpha: float = 0.001              # LASSO strength
    importance_threshold: Optional[float] = None  # SelectFromModel threshold

# ---------------- Top-level preprocessing -----------------
class Preprocessing(BaseModel):
    missing_values: ImputationConfig = ImputationConfig()
    outlier_removal: OutlierRemovalConfig = OutlierRemovalConfig()
    scaling: ScalingConfig = ScalingConfig()
    log_transform: LogTransformConfig = LogTransformConfig()
    batch_correction: BatchCorrectionConfig = BatchCorrectionConfig()
    qc_filtering: QCFilteringConfig = QCFilteringConfig()
    encoding: EncodingConfig = EncodingConfig()
    feature_selection: FeatureSelectionConfig = FeatureSelectionConfig()

# ---------------- Training config & API models -----------------
class Split(BaseModel):
    test_size: float = 0.2
    cv_folds: int = 5
    random_state: int = 42

class TrainConfig(BaseModel):
    target: str
    problem_type: Problem
    preprocessing: Preprocessing = Preprocessing()
    model: Literal[
        "random_forest",
        "svm",
        "neural_network",
        "gradient_boosting",
        "logistic_regression",
        "xgboost"
    ]
    hyperparams: Dict[str, Any] = {}
    split: Split = Split()

class TrainRequest(BaseModel):
    dataset_id: str
    dataset_uri: str
    config: TrainConfig

class TrainResponse(BaseModel):
    job_id: str

class JobStatus(BaseModel):
    id: str
    status: Literal["queued", "started", "finished", "failed"]
    metrics: Optional[Dict[str, Any]] = None
    model_path: Optional[str] = None

class PredictRequest(BaseModel):
    records: List[dict]

class PredictResponse(BaseModel):
    predictions: list
    probabilities: Optional[list] = None
