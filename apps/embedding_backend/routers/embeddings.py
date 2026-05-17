from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from apps.embedding_backend.auth.deps import get_current_user
from apps.embedding_backend.services.embedding_service import compute_embeddings
from apps.embedding_backend.workers.embedding_worker import run_embedding_job
from apps.embedding_backend.workers.queue_worker import get_queue
from storage.s3_storage import download_to_temp, is_s3_uri

router = APIRouter(prefix="/embeddings", tags=["embeddings"])


class EmbeddingRequest(BaseModel):
    drug_id: str = Field(..., min_length=1)
    canonical_smiles: str = Field(..., min_length=1)
    gene_id: str = Field(..., min_length=1)
    gene_sequence: str = Field(..., min_length=1)
    include_vectors: bool = True
    include_combined_csv: bool = True
    create_zip: bool = True


class EmbeddingAsyncResponse(BaseModel):
    job_id: str
    status: str
    message: str
    request_id: str


def _require_user_id(user: Dict[str, Any]) -> str:
    sub = user.get("sub")
    if not sub:
        raise HTTPException(401, "No subject in token")
    return str(sub)


def _verify_job_access(job, user_id: str) -> None:
    owner = str(job.meta.get("user_id", ""))
    if owner and owner != user_id:
        raise HTTPException(403, "Access denied")


@router.post("/sync")
async def generate_embeddings_sync(req: EmbeddingRequest, user=Depends(get_current_user)):
    user_id = _require_user_id(user)
    try:
        result = compute_embeddings(
            drug_id=req.drug_id,
            canonical_smiles=req.canonical_smiles,
            gene_id=req.gene_id,
            gene_sequence=req.gene_sequence,
            user_id=user_id,
            include_vectors=req.include_vectors,
            include_combined_csv=req.include_combined_csv,
            create_zip=req.create_zip,
        )
        return {
            "status": "finished",
            "request_id": result.request_id,
            "metadata": result.metadata,
            "dimensions": result.dimensions,
            "artifacts": result.artifacts,
            "vectors": result.vectors,
        }
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc


@router.post("/async", response_model=EmbeddingAsyncResponse)
async def generate_embeddings_async(req: EmbeddingRequest, user=Depends(get_current_user)):
    user_id = _require_user_id(user)
    q = get_queue()
    request_id = str(uuid.uuid4())
    payload = req.model_dump()
    payload["request_id"] = request_id

    job = q.enqueue(run_embedding_job, payload, user_id, job_timeout=45 * 60)
    job.meta["user_id"] = user_id
    job.meta["request_id"] = request_id
    job.save_meta()

    return {
        "job_id": job.id,
        "status": "queued",
        "message": "Embedding generation job queued.",
        "request_id": request_id,
    }


@router.get("/{job_id}/status")
async def get_embedding_status(job_id: str, user=Depends(get_current_user)):
    user_id = _require_user_id(user)
    q = get_queue()
    job = q.fetch_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    _verify_job_access(job, user_id)

    status = job.get_status()
    result = None
    if status == "finished":
        result = job.result
    elif status == "failed":
        result = {"error": job.exc_info}

    return {
        "job_id": job_id,
        "status": status,
        "request_id": job.meta.get("request_id"),
        "result": result,
    }


@router.get("/{job_id}/download")
async def download_embedding_artifacts(
    job_id: str,
    format: str = Query("zip", pattern="^(zip|metadata|drug|gene|combined)$"),
    user=Depends(get_current_user),
):
    user_id = _require_user_id(user)
    q = get_queue()
    job = q.fetch_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    _verify_job_access(job, user_id)
    if job.get_status() != "finished":
        raise HTTPException(400, "Job is not finished yet")
    if not isinstance(job.result, dict):
        raise HTTPException(500, "Job result is malformed")

    artifacts = job.result.get("artifacts", {})
    key_map = {
        "zip": "zip_file",
        "metadata": "input_metadata_csv",
        "drug": "drug_embeddings_csv",
        "gene": "gene_embeddings_csv",
        "combined": "combined_embeddings_csv",
    }
    artifact_key = key_map[format]
    artifact = artifacts.get(artifact_key)
    if not artifact:
        raise HTTPException(404, f"Artifact not available for format: {format}")

    if is_s3_uri(str(artifact)):
        file_path = await download_to_temp(str(artifact), suffix=".zip" if format == "zip" else ".csv")
    else:
        file_path = Path(artifact)
        if not file_path.exists():
            raise HTTPException(404, "Artifact file missing")

    media_type = "application/zip" if format == "zip" else "text/csv"
    return FileResponse(path=file_path, media_type=media_type, filename=file_path.name)

