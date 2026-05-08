from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from pathlib import Path
import os
from auth.deps import get_current_user
from client.db import db
from workers.queue_worker import get_queue
from typing import Optional, Dict, Any
router = APIRouter(prefix="/experiments", tags=["experiments"])

# Get artifacts directory - same logic as train_worker.py
ARTIFACTS_DIR = os.getenv("ARTIFACTS_DIR", "./artifacts")

@router.get("/{experiment_id}/genes/download")
async def download_ranked_genes_csv(experiment_id: str, user=Depends(get_current_user)):
    """
    Download the ranked-genes CSV produced during training for this experiment.
    The TrainingRun.resultsPath field stores the local/remote path.
    """
    if not user["sub"]:
        raise HTTPException(401, "No subject in token")

    experiment = await db.trainingrun.find_unique(where={"id": experiment_id})
    if not experiment:
        raise HTTPException(404, "Experiment not found")

    # Verify ownership
    if str(experiment.userId) != str(user["sub"]):
        raise HTTPException(403, "Access denied")

    results_path = getattr(experiment, "resultsPath", None)

    candidate_paths = []
    if results_path:
        p = Path(results_path)
        if not p.is_absolute():
            # Try relative to current working directory
            candidate_paths.append(Path.cwd() / p)
            # Also try relative to model_backend directory
            model_backend_dir = Path(__file__).resolve().parent.parent
            candidate_paths.append(model_backend_dir / p)
        else:
            candidate_paths.append(p)

    # Fallback: reconstruct expected artifacts path even if resultsPath was never saved
    # Use same logic as train_worker.py - ARTIFACTS_DIR defaults to "./artifacts"
    # Pattern: artifacts/<userId>/<job_id>/ranked_genes.csv
    artifacts_base = Path(ARTIFACTS_DIR)
    if not artifacts_base.is_absolute():
        # Resolve relative to current working directory (same as train_worker does)
        artifacts_base = Path.cwd() / artifacts_base
    
    derived_path = artifacts_base / str(experiment.userId) / experiment.id / "ranked_genes.csv"
    candidate_paths.append(derived_path)
    
    # Also try relative to model_backend directory (in case backend runs from project root)
    model_backend_dir = Path(__file__).resolve().parent.parent
    model_backend_artifacts = model_backend_dir / "artifacts" / str(experiment.userId) / experiment.id / "ranked_genes.csv"
    candidate_paths.append(model_backend_artifacts)

    path = None
    for p in candidate_paths:
        if p.exists():
            path = p
            break

    if path is None:
        # Preserve a clear error for the client
        raise HTTPException(404, "No ranked-genes CSV available for this experiment")

    return FileResponse(
        str(path),
        media_type="text/csv",
        filename=f"{experiment.id}_ranked_genes.csv",
    )


@router.get("")
async def list_experiments(user=Depends(get_current_user)):
    if not user["sub"]:
        raise HTTPException(401, "No subject in token")

    # Fetch experiments from the database
    experiments = await db.trainingrun.find_many(
        where={"userId": int(user["sub"])},
        order={"createdAt": "desc"}
    )

    return {"experiments": experiments}

@router.get("/{experiment_id}")
async def get_experiment_details(experiment_id: str, user=Depends(get_current_user)):
    if not user["sub"]:
        raise HTTPException(401, "No subject in token")

    # First check if job is still in Redis queue
    q = get_queue()
    job = q.fetch_job(experiment_id)
    
    # Get experiment from database
    experiment = await db.trainingrun.find_unique(
        where={"id": experiment_id}
    )
    
    if not experiment:
        raise HTTPException(404, "Experiment not found")
    
    # Verify ownership
    if str(experiment.userId) != str(user["sub"]):
        raise HTTPException(403, "Access denied")
    
    # Get current status from Redis if available
    current_status = experiment.status
    if job:
        redis_status = job.get_status(refresh=True)
        # Map Redis statuses to our statuses
        status_map = {
            "queued": "queued",
            "deferred": "queued",
            "started": "started",
            "finished": "finished",
            "failed": "failed"
        }
        current_status = status_map.get(redis_status, experiment.status)
    
    # Map status for UI compatibility
    status_map_ui = {
        "finished": "completed",
        "started": "running",
        "queued": "pending",
        "failed": "failed"
    }
    ui_status = status_map_ui.get(current_status, current_status)
    
    # Extract metrics
    metrics = experiment.metrics if experiment.metrics else {}
    if isinstance(metrics, dict):
        metrics = dict(metrics)
    
    # Build parameters from database parameters field (preferred) or fallback to job/metrics
    parameters = None
    problem_type = None
    
    # First try to get config from database parameters field
    if experiment.parameters:
        config_from_db = dict(experiment.parameters) if isinstance(experiment.parameters, dict) else experiment.parameters
        if isinstance(config_from_db, dict):
            preprocessing_steps = _extract_preprocessing_steps_from_config(config_from_db)
            problem_type = config_from_db.get("problem_type", "classification")
            parameters = {
                "model_type": config_from_db.get("model", "unknown"),
                "problem_type": problem_type,
                "num_folds": config_from_db.get("split", {}).get("cv_folds", 5),
                "train_test_split": config_from_db.get("split", {}).get("test_size", 0.2),
                "feature_selection": config_from_db.get("preprocessing", {}).get("feature_selection", {}).get("method") or None,
                "preprocessing_steps": preprocessing_steps,
                "hyperparameters": config_from_db.get("hyperparams", {}),
            }
    
    # Fallback to job args if database parameters not available
    if not parameters:
        config_from_job = None
        if job and hasattr(job, 'args') and job.args and len(job.args) >= 2:
            try:
                # job.args should be (dataset_uri, config, owner_id)
                config_from_job = job.args[1] if isinstance(job.args[1], dict) else None
            except Exception:
                pass
        
        if config_from_job:
            preprocessing_steps = _extract_preprocessing_steps_from_config(config_from_job)
            problem_type = config_from_job.get("problem_type", "classification")
            parameters = {
                "model_type": config_from_job.get("model", "unknown"),
                "problem_type": problem_type,
                "num_folds": config_from_job.get("split", {}).get("cv_folds", 5),
                "train_test_split": config_from_job.get("split", {}).get("test_size", 0.2),
                "feature_selection": config_from_job.get("preprocessing", {}).get("feature_selection", {}).get("method") or None,
                "preprocessing_steps": preprocessing_steps,
                "hyperparameters": config_from_job.get("hyperparams", {}),
            }
        elif metrics:
            # Last fallback to metrics (MLflow logged params)
            preprocessing_steps = _extract_preprocessing_steps(metrics)
            problem_type = metrics.get("problem_type", "classification")
            parameters = {
                "model_type": metrics.get("model") or metrics.get("model_type") or "unknown",
                "problem_type": problem_type,
                "num_folds": metrics.get("cv_folds") or metrics.get("cv_folds") or 5,
                "train_test_split": metrics.get("test_size") or metrics.get("split", {}).get("test_size") if isinstance(metrics.get("split"), dict) else 0.2,
                "feature_selection": metrics.get("feature_selection", {}).get("method") if isinstance(metrics.get("feature_selection"), dict) else None,
                "preprocessing_steps": preprocessing_steps if preprocessing_steps else [],
                "hyperparameters": metrics.get("hyperparams") or {},
            }
    
    # Build results from metrics
    results = None
    if metrics and current_status in ["finished", "failed"]:
        # Extract selected feature names for top_genes if available
        top_genes = []
        feature_selection_info = metrics.get("feature_selection")
        if isinstance(feature_selection_info, dict):
            selected_features = feature_selection_info.get("selected_feature_names", [])
            if selected_features and isinstance(selected_features, list):
                # Convert feature names to Gene-like objects.
                # We currently only know which features were selected, not their
                # per-gene statistics, so we leave expression/pvalue/foldChange
                # as null for the frontend to render as "N/A" rather than 0.
                top_genes = [
                    {
                        "symbol": str(feat),
                        "expression": None,
                        "pvalue": None,
                        "foldChange": None,
                    }
                    for feat in selected_features[:20]  # Limit to top 20
                ]
        
        # Determine problem type from parameters or default to classification
        problem_type = "classification"
        if parameters and parameters.get("problem_type"):
            problem_type = parameters.get("problem_type")
        elif metrics.get("problem_type"):
            problem_type = metrics.get("problem_type")
        
        results = {
            "problem_type": problem_type,
            # Classification metrics
            "accuracy": metrics.get("accuracy"),
            "precision_score": metrics.get("precision"),
            "recall_score": metrics.get("recall"),
            "f1_score": metrics.get("f1"),
            "roc_auc": metrics.get("roc_auc"),
            # Regression metrics
            "r2_score": metrics.get("r2"),
            "mse": metrics.get("mse"),
            "rmse": metrics.get("rmse"),
            # Common metrics
            "cv_mean": metrics.get("cv_mean"),
            "cv_std": metrics.get("cv_std"),
            "n_features_original": metrics.get("n_features_original"),
            "n_features_selected": metrics.get("n_features_selected"),
            "feature_selection": metrics.get("feature_selection"),
            "warnings": metrics.get("warnings"),
            "warnings_count": metrics.get("warnings_count"),
            "top_genes": top_genes,  # Always include, even if empty
            "additional_metrics": {k: v for k, v in metrics.items() 
                                 if k not in ["accuracy", "precision", "recall", "f1", "roc_auc", 
                                             "r2", "mse", "rmse", "cv_mean", "cv_std",
                                             "n_features_original", "n_features_selected",
                                             "feature_selection", "warnings", "warnings_count", "problem_type"]},
        }
    
    # Extract errors if failed
    errors = None
    if current_status == "failed":
        if metrics:
            errors = {
                "error": metrics.get("error") or metrics.get("fit_error") or metrics.get("cv_error"),
                "traceback": metrics.get("traceback"),
                "warnings": metrics.get("warnings", []),
            }
        elif job and job.result:
            errors = {
                "error": str(job.result.get("error", "Unknown error")),
            }
    
    return {
        "experiment": {
            "id": experiment.id,
            "user_id": str(experiment.userId),
            "name": experiment.name or f"Experiment {experiment.id[:8]}",
            "description": experiment.description or "",
            "status": ui_status,  # Use UI-compatible status
            "createdAt": experiment.createdAt.isoformat() if experiment.createdAt else None,
            "updatedAt": experiment.updatedAt.isoformat() if experiment.updatedAt else None,
            "datasetUri": experiment.datasetUri,
            "modelPath": experiment.modelPath,
            "resultsPath": getattr(experiment, "resultsPath", None),
        },
        "parameters": parameters,
        "results": results,
        "errors": errors,
    }

def _extract_preprocessing_steps(metrics: Dict[str, Any]) -> list:
    """Extract preprocessing steps from metrics/config"""
    steps = []
    
    # Check for preprocessing config in metrics
    prep_config = metrics.get("preprocessing") or {}
    
    # Missing value imputation: only show as an explicit step if the user has
    # configured something beyond the safe defaults (e.g. dropping rows or
    # specifying custom fill values/strategies).
    mv_cfg = prep_config.get("missing_values", {}) or {}
    if (
        mv_cfg.get("drop_rows")
        or mv_cfg.get("fill_value_numeric") is not None
        or mv_cfg.get("fill_value_categorical") is not None
        or mv_cfg.get("strategy_numeric") not in (None, "median")
        or mv_cfg.get("strategy_categorical") not in (None, "most_frequent")
    ):
        steps.append("Missing Value Imputation")
    if prep_config.get("scaling", {}).get("method") and prep_config.get("scaling", {}).get("method") != "none":
        steps.append("Scaling")
    if prep_config.get("log_transform", {}).get("enabled"):
        steps.append("Log Transform")
    if prep_config.get("outlier_removal", {}).get("method") and prep_config.get("outlier_removal", {}).get("method") != "none":
        steps.append("Outlier Removal")
    if prep_config.get("batch_correction", {}).get("enabled"):
        steps.append("Batch Correction")
    if prep_config.get("qc_filtering", {}).get("enabled"):
        steps.append("QC Filtering")
    if prep_config.get("encoding", {}).get("method") and prep_config.get("encoding", {}).get("method") != "none":
        steps.append("Encoding")
    if prep_config.get("feature_selection", {}).get("method") and prep_config.get("feature_selection", {}).get("method") != "none":
        steps.append("Feature Selection")
    
    return steps

def _extract_preprocessing_steps_from_config(config: Dict[str, Any]) -> list:
    """Extract preprocessing steps from training config"""
    steps = []
    
    prep_config = config.get("preprocessing", {})
    
    # Missing value imputation: only show when configured beyond defaults
    mv_cfg = prep_config.get("missing_values", {}) or {}
    if (
        mv_cfg.get("drop_rows")
        or mv_cfg.get("fill_value_numeric") is not None
        or mv_cfg.get("fill_value_categorical") is not None
        or mv_cfg.get("strategy_numeric") not in (None, "median")
        or mv_cfg.get("strategy_categorical") not in (None, "most_frequent")
    ):
        steps.append("Missing Value Imputation")
    if prep_config.get("scaling", {}).get("method") and prep_config.get("scaling", {}).get("method") != "none":
        steps.append("Scaling")
    if prep_config.get("log_transform", {}).get("enabled"):
        steps.append("Log Transform")
    if prep_config.get("outlier_removal", {}).get("method") and prep_config.get("outlier_removal", {}).get("method") != "none":
        steps.append("Outlier Removal")
    if prep_config.get("batch_correction", {}).get("enabled"):
        steps.append("Batch Correction")
    if prep_config.get("qc_filtering", {}).get("enabled"):
        steps.append("QC Filtering")
    if prep_config.get("encoding", {}).get("method") and prep_config.get("encoding", {}).get("method") != "none":
        steps.append("Encoding")
    if prep_config.get("feature_selection", {}).get("method") and prep_config.get("feature_selection", {}).get("method") != "none":
        steps.append("Feature Selection")
    
    return steps