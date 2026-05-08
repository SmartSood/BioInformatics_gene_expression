from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional, Dict
from pathlib import Path
import os
import sys
from pathlib import Path as PathLib

# Add model_backend to path to access database
PROJECT_ROOT = PathLib(__file__).resolve().parents[3]
MODEL_BACKEND_DIR = PROJECT_ROOT / "apps" / "model_backend"
if str(MODEL_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_BACKEND_DIR))

from apps.depmap_backend.auth.deps import get_current_user
from apps.depmap_backend.workers.queue_worker import get_queue
from apps.depmap_backend.workers.depmap_worker import run_depmap_association
from client.db import db, connect_db
from prisma import Json
import logging

logger = logging.getLogger("depmap_backend.associations")

router = APIRouter(prefix="/associations", tags=["associations"])


class GeneAssociationRequest(BaseModel):
    genes: List[str]  # List of gene symbols, e.g., ["ERCC3", "TP53"]
    experiment_id: Optional[str] = None  # Optional experiment ID for organizing outputs
    force: bool = False  # Force regeneration even if file exists


class GeneAssociationResponse(BaseModel):
    job_id: str
    status: str
    message: str
    existing_files: Optional[Dict[str, str]] = None


@router.post("")
async def create_gene_association(
    req: GeneAssociationRequest,
    user=Depends(get_current_user)
):
    """
    Create a new gene-drug association analysis job.
    Checks database first to see if results already exist (unless force=true).
    Returns a job_id that can be used to check status and download results.
    """
    if not user["sub"]:
        raise HTTPException(401, "No subject in token")

    if not req.genes:
        raise HTTPException(400, "At least one gene must be provided")

    # Strip whitespace from gene names and normalize (uppercase for consistency)
    cleaned_genes = [g.strip().upper() for g in req.genes if g.strip()]
    if not cleaned_genes:
        raise HTTPException(400, "No valid genes provided after trimming whitespace")

    # Get user ID and experiment ID
    user_id = str(user["sub"])
    experiment_id = req.experiment_id or "default"

    # Check database for existing results (unless force=true)
    if not req.force and experiment_id != "default":
        try:
            logger.info(f"=== Checking database for experiment {experiment_id}, genes: {cleaned_genes}, force: {req.force} ===")
            # Ensure database is connected
            await connect_db()
            
            experiment = await db.trainingrun.find_unique(where={"id": experiment_id})
            logger.info(f"Experiment found: {experiment is not None}")
            
            if experiment:
                # Check if field exists (might not exist if migration not run)
                has_field = hasattr(experiment, 'geneDepmapResults')
                has_value = has_field and experiment.geneDepmapResults is not None
                logger.info(f"Has geneDepmapResults field: {has_field}, Has value: {has_value}")
                
                if has_field and has_value:
                    depmap_results = dict(experiment.geneDepmapResults) if isinstance(experiment.geneDepmapResults, dict) else {}
                    logger.info(f"Raw depmap_results type: {type(experiment.geneDepmapResults)}, keys: {list(depmap_results.keys()) if depmap_results else 'empty'}")
                    logger.info(f"Requested genes (normalized): {cleaned_genes}")
                    
                    # Normalize keys in depmap_results for case-insensitive matching
                    normalized_depmap_results = {}
                    for key, value in depmap_results.items():
                        normalized_key = str(key).strip().upper()
                        normalized_depmap_results[normalized_key] = value
                        logger.info(f"  Normalized key: '{key}' -> '{normalized_key}' -> file: {value}")
                    
                    logger.info(f"Normalized depmap_results keys: {list(normalized_depmap_results.keys())}")
                    
                    # Check if all requested genes already have results
                    missing_genes = []
                    existing_files = {}
                    for gene in cleaned_genes:
                        logger.info(f"Checking gene: '{gene}'")
                        if gene in normalized_depmap_results:
                            file_path = Path(normalized_depmap_results[gene])
                            logger.info(f"  Found in DB, checking file: {file_path}")
                            logger.info(f"  File exists: {file_path.exists()}")
                            if file_path.exists():
                                existing_files[gene] = str(file_path)
                                logger.info(f"  ✓ Found existing file for gene {gene}: {file_path}")
                            else:
                                missing_genes.append(gene)
                                logger.warning(f"  ✗ File for gene {gene} doesn't exist: {file_path}")
                        else:
                            missing_genes.append(gene)
                            logger.warning(f"  ✗ Gene '{gene}' not found in depmap_results. Available keys: {list(normalized_depmap_results.keys())}")
                    
                    # If all genes have existing files, return them immediately
                    if not missing_genes and existing_files:
                        logger.info(f"✓✓✓ Returning cached results for genes: {', '.join(cleaned_genes)}")
                        return {
                            "job_id": "cached",
                            "status": "finished",
                            "message": f"Results already exist for genes: {', '.join(cleaned_genes)}. Use force=true to regenerate.",
                            "existing_files": existing_files
                        }
                    
                    # If some genes are missing, only process those
                    if missing_genes and existing_files:
                        logger.info(f"Processing only missing genes: {missing_genes}")
                        cleaned_genes = missing_genes  # Only process missing genes
                else:
                    logger.info(f"Experiment {experiment_id} has no geneDepmapResults field or it's empty (has_field={has_field}, has_value={has_value})")
            else:
                logger.warning(f"Experiment {experiment_id} not found in database")
        except Exception as e:
            # If database check fails, log and continue with normal processing
            logger.error(f"Failed to check database for existing results: {e}")
            import traceback
            logger.error(traceback.format_exc())

    # Enqueue job
    q = get_queue()
    job = q.enqueue(
        run_depmap_association,
        cleaned_genes,
        user_id,
        experiment_id,
        req.force,
        job_timeout=30 * 60,  # 30 minutes timeout
    )

    return {
        "job_id": job.id,
        "status": "queued",
        "message": f"Association analysis queued for genes: {', '.join(cleaned_genes)}"
    }


@router.get("/{job_id}/status")
async def get_association_status(
    job_id: str,
    user=Depends(get_current_user)
):
    """
    Get the status of a gene association job.
    """
    if not user["sub"]:
        raise HTTPException(401, "No subject in token")

    q = get_queue()
    job = q.fetch_job(job_id)

    if not job:
        raise HTTPException(404, "Job not found")

    # Get job result if available
    status = job.get_status()
    result = None
    if status == "finished":
        try:
            result = job.result
        except Exception:
            pass

    return {
        "job_id": job_id,
        "status": status,
        "result": result
    }


@router.get("/experiment/{experiment_id}/gene/{gene_name}/download")
async def download_association_by_gene(
    experiment_id: str,
    gene_name: str,
    user=Depends(get_current_user)
):
    """
    Download CSV file for a specific gene from an experiment.
    Uses the database to find the file path.
    """
    if not user["sub"]:
        raise HTTPException(401, "No subject in token")

    try:
        # Ensure database is connected
        await connect_db()
        
        experiment = await db.trainingrun.find_unique(where={"id": experiment_id})
        if not experiment:
            raise HTTPException(404, "Experiment not found")
        
        # Verify ownership
        if str(experiment.userId) != str(user["sub"]):
            raise HTTPException(403, "Access denied")
        
        # Get file path from database
        if not hasattr(experiment, 'geneDepmapResults') or not experiment.geneDepmapResults:
            raise HTTPException(404, "No DepMap results found for this experiment")
        
        depmap_results = dict(experiment.geneDepmapResults) if isinstance(experiment.geneDepmapResults, dict) else {}
        # Normalize gene name to uppercase for case-insensitive matching
        gene_name_clean = gene_name.strip().upper()
        
        # Normalize keys in depmap_results for case-insensitive matching
        normalized_depmap_results = {}
        for key, value in depmap_results.items():
            normalized_key = str(key).strip().upper()
            normalized_depmap_results[normalized_key] = value
        
        logger.info(f"Looking for gene {gene_name_clean} in normalized results: {list(normalized_depmap_results.keys())}")
        
        if gene_name_clean not in normalized_depmap_results:
            raise HTTPException(404, f"No DepMap results found for gene: {gene_name_clean}. Available genes: {list(normalized_depmap_results.keys())}")
        
        csv_path = Path(normalized_depmap_results[gene_name_clean])
        if not csv_path.exists():
            raise HTTPException(404, "CSV file not found on server")
        
        return FileResponse(
            str(csv_path),
            media_type="text/csv",
            filename=f"depmap_{gene_name_clean}_associations.csv",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading by gene: {e}")
        raise HTTPException(500, f"Error downloading file: {str(e)}")


@router.get("/{job_id}/download")
async def download_association_csv(
    job_id: str,
    user=Depends(get_current_user)
):
    """
    Download the CSV file with gene-drug associations.
    Verifies that the user owns the job before allowing download.
    """
    if not user["sub"]:
        raise HTTPException(401, "No subject in token")

    # Handle cached results
    if job_id == "cached":
        raise HTTPException(400, "Use /experiment/{experiment_id}/gene/{gene_name}/download for cached results")

    q = get_queue()
    job = q.fetch_job(job_id)

    if not job:
        raise HTTPException(404, "Job not found")

    # Verify job was created by this user by checking job args
    # Job args: (genes, user_id, experiment_id, force)
    if len(job.args) >= 2 and str(job.args[1]) != str(user["sub"]):
        raise HTTPException(403, "Access denied: You don't own this job")

    if job.get_status() != "finished":
        raise HTTPException(400, f"Job is not finished. Current status: {job.get_status()}")

    # Get result path from job result
    result = job.result
    if not result or "csv_path" not in result:
        raise HTTPException(404, "CSV file not found in job result")

    csv_path = Path(result["csv_path"])
    if not csv_path.exists():
        raise HTTPException(404, "CSV file not found on server")

    # Additional security: verify path contains user ID
    user_id = str(user["sub"])
    if user_id not in str(csv_path):
        raise HTTPException(403, "Access denied: Invalid file path")

    return FileResponse(
        str(csv_path),
        media_type="text/csv",
        filename=f"depmap_associations_{job_id}.csv",
    )

