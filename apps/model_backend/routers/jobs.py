from fastapi import APIRouter, Depends
from pathlib import Path
import os

from auth.deps import get_current_user
from scemma.model import JobStatus
from workers.queue_worker import get_queue
from client.db import db

router = APIRouter(prefix="/jobs", tags=["jobs"])

ARTIFACTS_DIR = os.getenv("ARTIFACTS_DIR", "./artifacts")
MODEL_FILENAME = "model.joblib"

def _fs_model_path(user_id: str, job_id: str) -> str | None:
    p = Path(ARTIFACTS_DIR) / str(user_id) / job_id / MODEL_FILENAME
    return str(p) if p.exists() else None

@router.get("/{job_id}", response_model=JobStatus)
async def job_status(job_id: str, user=Depends(get_current_user)):
    q = get_queue()
    job = q.fetch_job(job_id)

    if job is not None:
        status = job.get_status(refresh=True)

        if status in {"queued", "deferred"}:
            return {"id": job_id, "status": "queued", "metrics": None, "model_path": None}
        if status == "started":
            return {"id": job_id, "status": "started", "metrics": None, "model_path": None}
        if status == "finished":
            res = job.result or {}
            return {
                "id": job_id,
                "status": "finished",
                "metrics": res.get("metrics"),
                "model_path": res.get("model_path") or _fs_model_path(user["sub"], job_id),
            }
        if status == "failed":
            return {
                "id": job_id,
                "status": "failed",
                "metrics": None,
                "model_path": _fs_model_path(user["sub"], job_id),
            }

    # Not present in Redis → ask DB
    run = await db.trainingrun.find_unique(where={"id": job_id})
    if run and str(run.userId) == str(user["sub"]):
        return {
            "id": job_id,
            "status": run.status,
            "metrics": run.metrics,
            "model_path": run.modelPath or _fs_model_path(user["sub"], job_id),
        }

    # Fallback to filesystem
    mp = _fs_model_path(user["sub"], job_id)
    if mp:
        return {"id": job_id, "status": "finished", "metrics": None, "model_path": mp}

    return {"id": job_id, "status": "not_found", "metrics": None, "model_path": None}
