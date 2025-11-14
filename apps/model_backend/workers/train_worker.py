# workers/train_worker.py
import os
from pathlib import Path
import asyncio
from typing import Optional, Any, Dict
from rq import get_current_job
import logging
import json
import joblib
import traceback

# DB helpers
from client.db import db, connect_db
from workers.db_utils import sanitize_metrics, update_trainingrun_with_retries

# Optional numpy/pandas lazy imports
try:
    import numpy as np
except Exception:
    np = None

try:
    import pandas as pd
except Exception:
    pd = None

ARTIFACTS_DIR = os.getenv("ARTIFACTS_DIR", "./artifacts")

# logger: reuse train_worker logger (or create if missing)
logger = logging.getLogger("train_worker")
logger.setLevel(logging.INFO)
if not logger.handlers:
    # If no handlers configured, add a simple file handler
    fh = logging.FileHandler("worker_debug.log")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
    logger.addHandler(fh)


def _try_load_file(path: Path) -> Any:
    """Try to load a single metric file with common loaders."""
    # numpy .npy/.npz
    try:
        if np is not None:
            val = np.load(str(path), allow_pickle=True)
            # zero-dim numpy scalar -> python scalar
            if isinstance(val, np.ndarray) and val.shape == ():
                return val.item()
            return val
    except Exception:
        pass

    # joblib
    try:
        val = joblib.load(str(path))
        return val
    except Exception:
        pass

    # try JSON text
    try:
        txt = path.read_text(encoding="utf-8")
        return json.loads(txt)
    except Exception:
        pass

    # numeric or plain text
    try:
        txt = path.read_text(encoding="utf-8").strip()
        if txt == "":
            return None
        try:
            return int(txt)
        except Exception:
            pass
        try:
            return float(txt)
        except Exception:
            pass
        return txt
    except Exception:
        pass

    # last-resort: bytes
    try:
        return path.read_bytes()
    except Exception:
        return None


def load_metrics_from_artifacts(out_dir: Path) -> Dict[str, Any]:
    metrics_dir = out_dir / "metrics"
    if not metrics_dir.exists() or not metrics_dir.is_dir():
        return {}
    metrics = {}
    for child in sorted(metrics_dir.iterdir()):
        if child.is_dir():
            continue
        key = child.stem  # filename without extension
        try:
            val = _try_load_file(child)
            # convert numpy arrays to lists
            if np is not None and isinstance(val, np.ndarray):
                try:
                    val = val.tolist()
                except Exception:
                    val = [x.item() if isinstance(x, np.generic) else x for x in val]
        except Exception as e:
            logger.warning("Failed to load metric file %s: %s", child, e)
            val = None
        metrics[key] = val
    return metrics


def _coerce_metrics_for_prisma(metrics: Any) -> Optional[Any]:
    """
    Ensure metrics is JSON-native and Prisma-acceptable.
    Returns sanitized value or None if coercion fails.
    """
    try:
        sanitized = sanitize_metrics(metrics)  # your robust sanitizer
        # Force JSON round-trip to convert any exotic types to pure python primitives/structures
        dumped = json.dumps(sanitized, default=str)
        coerced = json.loads(dumped)
        return coerced
    except Exception as exc:
        logger.warning("Metrics coercion failed: %s. metrics repr truncated: %s", exc, repr(metrics)[:1000])
        return None


def run_train(dataset_uri: str, config: dict, owner_id: str):
    job = get_current_job()
    job_id: Optional[str] = job.id if job else "no_jobid"
    job_id = str(job_id)

    out_dir = Path(ARTIFACTS_DIR) / str(owner_id) / job_id
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = dataset_uri
    model_path = str(out_dir / "model.joblib")

    # defer heavy import
    from pipeline.pipeline import train

    async def _run():
        prisma = None
        try:
            try:
                prisma = await connect_db()
            except Exception as e:
                prisma = None
                logger.warning("Warning: failed to connect to DB at job start: %s", e)

            if prisma:
                try:
                    await prisma.trainingrun.upsert(
                        where={"id": job_id},
                        data={
                            "create": {
                                "id": job_id,
                                "userId": int(owner_id) if str(owner_id).isdigit() else owner_id,
                                "status": "started",
                                "datasetUri": dataset_path,
                            },
                            "update": {"status": "started"},
                        },
                    )
                except Exception as e:
                    logger.warning("Warning: upsert(started) failed: %s", e)

            # synchronous train function
            try:
                result = train(dataset_path, config, str(out_dir))
            except Exception as e:
                # Capture the error and traceback
                error_traceback = traceback.format_exc()
                error_msg = f"Training failed: {str(e)}\n{error_traceback}"
                logger.error("Training error for job %s: %s", job_id, error_msg)
                
                # Store error in result
                result = {
                    "error": str(e),
                    "error_traceback": error_traceback,
                    "model_path": None,
                    "metrics": None,
                    "warnings": None
                }
                
                # Update DB with failed status
                if prisma:
                    try:
                        await update_trainingrun_with_retries(
                            prisma, 
                            job_id, 
                            {
                                "status": "failed",
                                "metrics": _coerce_metrics_for_prisma({"error": str(e), "error_traceback": error_traceback})
                            }, 
                            attempts=3, 
                            base_delay=0.5
                        )
                    except Exception:
                        logger.exception("Failed to update DB with error status for job %s", job_id)
                
                # Re-raise to mark job as failed in RQ
                raise ValueError(error_msg) from e
            
            # Get model_path from result (this is the actual path where model was saved)
            actual_model_path = (result or {}).get("model_path") or model_path
            
            # First try to get metrics from result
            raw_metrics = (result or {}).get("metrics")
            metrics = None

            if isinstance(raw_metrics, dict):
                metrics = raw_metrics
                # Ensure feature selection info is preserved if it exists
                if "feature_selection" in metrics:
                    feature_selection_info = metrics.get("feature_selection")
                    # Ensure top-level counts are present
                    if "n_features_original" not in metrics and feature_selection_info:
                        metrics["n_features_original"] = feature_selection_info.get("n_features_original")
                    if "n_features_selected" not in metrics and feature_selection_info:
                        metrics["n_features_selected"] = feature_selection_info.get("n_features_selected")
            elif raw_metrics is not None:
                # if it's path-like string pointing into artifacts, try to interpret
                try:
                    # If it's a path to the artifact folder or file
                    p = Path(str(raw_metrics))
                    if p.exists():
                        # If raw_metrics points to a file, try load; if folder, look for metrics dir inside
                        if p.is_file():
                            metrics = _try_load_file(p)
                        elif p.is_dir():
                            # look for a metrics/ inside that dir
                            metrics = load_metrics_from_artifacts(p)
                    else:
                        # not a path; keep raw value (sanitize will handle if possible)
                        metrics = raw_metrics
                except Exception:
                    metrics = raw_metrics
            else:
                # If result didn't return metrics, try to load from out_dir/artifacts/metrics
                try:
                    artifact_metrics = load_metrics_from_artifacts(out_dir)
                    if artifact_metrics:
                        metrics = artifact_metrics
                    else:
                        metrics = None
                except Exception:
                    metrics = None

            # Include warnings in metrics if present
            warnings_list = (result or {}).get("warnings")
            if warnings_list:
                if not metrics:
                    metrics = {}
                metrics["warnings"] = warnings_list
                metrics["warnings_count"] = len(warnings_list)

            # Include feature selection info if present
            feature_selection_info = (result or {}).get("feature_selection")
            if feature_selection_info:
                if not metrics:
                    metrics = {}
                # Merge feature selection info into metrics
                metrics["feature_selection"] = feature_selection_info
                # Ensure top-level counts are present for easy access
                if "n_features_original" not in metrics:
                    metrics["n_features_original"] = feature_selection_info.get("n_features_original")
                if "n_features_selected" not in metrics:
                    metrics["n_features_selected"] = feature_selection_info.get("n_features_selected")
            
            # Also check if feature counts are in metrics directly from result
            if not metrics:
                metrics = {}
            if "n_features_original" not in metrics:
                metrics["n_features_original"] = result.get("n_features_original") if result else None
            if "n_features_selected" not in metrics:
                metrics["n_features_selected"] = result.get("n_features_selected") if result else None

            # Coerce to JSON-safe structure for Prisma
            metrics_clean = _coerce_metrics_for_prisma(metrics)

            # Prepare payload and update DB (awaiting the async update helper)
            if prisma:
                payload = {
                    "status": "finished",
                    "modelPath": actual_model_path,
                    "metrics": metrics_clean,
                }
                # Log the payload truncated (so you can inspect in logs)
                try:
                    payload_json = json.dumps(payload, default=str)
                except Exception:
                    payload_json = repr(payload)
                logger.info("Attempting final DB update for job %s; payload (truncated): %s", job_id, payload_json[:2000])

                try:
                    await update_trainingrun_with_retries(prisma, job_id, payload, attempts=3, base_delay=0.5)
                    logger.info("DB updated to finished for job %s", job_id)
                except Exception:
                    logger.exception("CRITICAL: final DB update failed for job %s — see traceback below:", job_id)
                    # re-raise so RQ marks job failed (optional)
                    raise

            enriched = dict(result or {})
            enriched.setdefault("job_id", job_id)
            enriched.setdefault("model_path", actual_model_path)
            enriched.setdefault("metrics", metrics)
            return enriched

        except Exception:
            # on exception, best-effort mark failed
            if prisma:
                try:
                    try:
                        await update_trainingrun_with_retries(prisma, job_id, {"status": "failed"}, attempts=2, base_delay=0.2)
                    except Exception:
                        logger.warning("Warning: marking job failed in DB also failed for job %s", job_id)
                except Exception:
                    pass
            raise

    return asyncio.run(_run())
