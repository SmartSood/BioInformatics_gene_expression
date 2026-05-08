# apps/depmap_backend/workers/depmap_worker.py
"""
Worker function for processing DepMap gene-drug associations.
This worker loads datasets once (caching) and processes gene associations.
"""
import os
import sys
from pathlib import Path
from typing import List, Dict, Any
import logging
import traceback

# Add depmap directory to path to import the associations script
DEPMAP_DIR = Path(__file__).resolve().parents[2] / "depmap"
if str(DEPMAP_DIR) not in sys.path:
    sys.path.insert(0, str(DEPMAP_DIR))

# Import functions from depmap_associations.py
from depmap_associations import (
    _load_expression,
    _load_model_metadata,
    _load_gdsc_auc,
    _load_ctrp_auc,
    _load_prism_matrix,
    _load_prism_public_24q2,
    _compute_correlations,
    DEFAULT_EXPRESSION_FILE,
    DEFAULT_MODEL_FILE,
    DEFAULT_GDSC_DOSE_RESPONSE_FILE,
    DEFAULT_CTRP_ZIP_FILE,
    DEFAULT_PRISM_SECONDARY_MATRIX_FILE,
    DEFAULT_PRISM_SECONDARY_TREATMENT_INFO_FILE,
    DEFAULT_PRISM_SECONDARY_CELL_INFO_FILE,
    DEFAULT_PRISM_PUBLIC_MATRIX_FILE,
    DEFAULT_PRISM_PUBLIC_TREATMENT_INFO_FILE,
    DEFAULT_PRISM_PUBLIC_CELL_INFO_FILE,
)

import pandas as pd
import csv

logger = logging.getLogger("depmap_worker")

# Global cache for loaded datasets (loaded once per worker process)
_cache = {
    "model": None,
    "gdsc": None,
    "ctrp": None,
    "prism_secondary": None,
    "prism_public": None,
}


def _get_dataset_path(relative_path: str) -> Path:
    """Get absolute path to dataset file."""
    depmap_dir = Path(__file__).resolve().parents[2] / "depmap"
    return depmap_dir / relative_path


def _load_cached_model() -> pd.DataFrame:
    """Load and cache model metadata."""
    if _cache["model"] is None:
        model_path = _get_dataset_path(DEFAULT_MODEL_FILE)
        logger.info(f"Loading model metadata from {model_path}")
        _cache["model"] = _load_model_metadata(str(model_path))
    return _cache["model"]


def run_depmap_association(genes: List[str], user_id: str, experiment_id: str, force: bool = False) -> Dict[str, Any]:
    """
    Process gene-drug associations for the given genes.
    
    Args:
        genes: List of gene symbols (e.g., ["ERCC3", "TP53"])
        user_id: User ID for organizing output files
        experiment_id: Experiment ID for organizing output files
        force: If True, regenerate even if file exists
    
    Returns:
        Dict with 'csv_path' and 'gene_count'
    """
    from rq import get_current_job
    import asyncio
    
    job = get_current_job()
    job_id = job.id if job else "unknown"
    
    # Strip whitespace from all gene names and normalize (uppercase for consistency)
    genes = [g.strip().upper() for g in genes if g.strip()]
    
    if not genes:
        raise ValueError("No valid genes provided after trimming whitespace")
    
    try:
        logger.info(f"Starting DepMap association analysis for genes: {', '.join(genes)} (user: {user_id}, experiment: {experiment_id}, force: {force})")
        
        # Get dataset paths
        expression_path = _get_dataset_path(DEFAULT_EXPRESSION_FILE)
        model_path = _get_dataset_path(DEFAULT_MODEL_FILE)
        gdsc_path = _get_dataset_path(DEFAULT_GDSC_DOSE_RESPONSE_FILE)
        ctrp_path = _get_dataset_path(DEFAULT_CTRP_ZIP_FILE)
        prism_secondary_matrix = _get_dataset_path(DEFAULT_PRISM_SECONDARY_MATRIX_FILE)
        prism_secondary_treatment = _get_dataset_path(DEFAULT_PRISM_SECONDARY_TREATMENT_INFO_FILE)
        prism_secondary_cell = _get_dataset_path(DEFAULT_PRISM_SECONDARY_CELL_INFO_FILE)
        prism_public_matrix = _get_dataset_path(DEFAULT_PRISM_PUBLIC_MATRIX_FILE)
        prism_public_treatment = _get_dataset_path(DEFAULT_PRISM_PUBLIC_TREATMENT_INFO_FILE)
        prism_public_cell = _get_dataset_path(DEFAULT_PRISM_PUBLIC_CELL_INFO_FILE)
        
        # Check required files
        if not expression_path.exists():
            raise FileNotFoundError(f"Expression file not found: {expression_path}")
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load expression for genes
        logger.info(f"Loading expression for genes: {', '.join(genes)}")
        expression = _load_expression(str(expression_path), genes)
        
        # Load model metadata (cached)
        logger.info("Loading model metadata")
        model = _load_cached_model()
        
        all_results = []
        
        # Process GDSC
        if gdsc_path.exists():
            logger.info("Loading GDSC dose-response data...")
            if _cache["gdsc"] is None:
                _cache["gdsc"] = _load_gdsc_auc(str(gdsc_path), model)
            
            for gdsc, dataset_label in _cache["gdsc"]:
                logger.info(f"Computing expression–{dataset_label} associations...")
                gdsc_res = _compute_correlations(expression, gdsc, dataset_label=dataset_label)
                all_results.append(gdsc_res)
        else:
            logger.warning("GDSC file not found, skipping")
        
        # Process CTRP
        if ctrp_path.exists():
            logger.info("Loading CTRP curves...")
            if _cache["ctrp"] is None:
                _cache["ctrp"] = _load_ctrp_auc(str(ctrp_path), model)
            
            logger.info("Computing expression–CTRP associations...")
            ctrp_res = _compute_correlations(
                expression, _cache["ctrp"], dataset_label="Drug sensitivity AUC (CTD^2)"
            )
            all_results.append(ctrp_res)
        else:
            logger.warning("CTRP file not found, skipping")
        
        # Process PRISM Secondary
        if all(p.exists() for p in [prism_secondary_matrix, prism_secondary_treatment, prism_secondary_cell]):
            logger.info("Loading PRISM Secondary Screen data...")
            if _cache["prism_secondary"] is None:
                _cache["prism_secondary"] = _load_prism_matrix(
                    str(prism_secondary_matrix),
                    str(prism_secondary_treatment),
                    str(prism_secondary_cell),
                    model,
                )
            
            logger.info("Computing expression–PRISM Secondary Screen associations...")
            prism_secondary_res = _compute_correlations(
                expression,
                _cache["prism_secondary"],
                dataset_label="Drug sensitivity AUC (PRISM Repurposing Secondary Screen)",
            )
            all_results.append(prism_secondary_res)
        else:
            logger.warning("PRISM Secondary files not found, skipping")
        
        # Process PRISM Public
        if all(p.exists() for p in [prism_public_matrix, prism_public_treatment, prism_public_cell]):
            logger.info("Loading PRISM Public 24Q2 data...")
            if _cache["prism_public"] is None:
                _cache["prism_public"] = _load_prism_public_24q2(
                    str(prism_public_matrix),
                    str(prism_public_treatment),
                    str(prism_public_cell),
                    model,
                )
            
            logger.info("Computing expression–PRISM Public associations...")
            prism_public_res = _compute_correlations(
                expression,
                _cache["prism_public"],
                dataset_label="PRISM Repurposing Public 24Q2",
            )
            all_results.append(prism_public_res)
        else:
            logger.warning("PRISM Public files not found, skipping")
        
        if not all_results:
            raise ValueError("No drug sensitivity datasets were loaded; nothing to compute.")
        
        # Combine results
        combined = pd.concat(all_results, ignore_index=True)
        combined = combined.sort_values(
            "Correlation", key=lambda s: s.abs(), ascending=False
        )
        
        # Save CSV organized by userId/experimentId
        output_dir = Path(__file__).resolve().parents[2] / "depmap_backend" / "outputs" / str(user_id) / str(experiment_id)
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / f"{job_id}_associations.csv"
        
        combined.to_csv(csv_path, index=False, quoting=csv.QUOTE_MINIMAL)
        logger.info(f"Wrote associations for {len(genes)} gene(s) to {csv_path}")
        
        # Save file locations to database (if experiment_id is valid)
        if experiment_id != "default":
            try:
                # Import database client
                import sys
                from pathlib import Path as PathLib
                PROJECT_ROOT = PathLib(__file__).resolve().parents[3]
                MODEL_BACKEND_DIR = PROJECT_ROOT / "apps" / "model_backend"
                if str(MODEL_BACKEND_DIR) not in sys.path:
                    sys.path.insert(0, str(MODEL_BACKEND_DIR))
                
                from client.db import db, connect_db
                from prisma import Json
                
                # Run async database update (similar to train_worker pattern)
                async def update_database():
                    prisma = None
                    try:
                        logger.info(f"=== Updating database for experiment {experiment_id} ===")
                        prisma = await connect_db()
                        if not prisma:
                            logger.warning("Failed to connect to database")
                            return
                            
                        experiment = await prisma.trainingrun.find_unique(where={"id": experiment_id})
                        if not experiment:
                            logger.warning(f"Experiment {experiment_id} not found in database")
                            return
                        
                        existing_results = {}
                        if hasattr(experiment, 'geneDepmapResults') and experiment.geneDepmapResults:
                            if isinstance(experiment.geneDepmapResults, dict):
                                # Normalize existing keys to uppercase for consistency
                                for key, value in dict(experiment.geneDepmapResults).items():
                                    normalized_key = str(key).strip().upper()
                                    existing_results[normalized_key] = value
                                logger.info(f"Existing results before update: {list(existing_results.keys())}")
                        
                        # Update with new gene file locations (genes are already normalized to uppercase)
                        for gene in genes:
                            existing_results[gene] = str(csv_path)
                            logger.info(f"Adding/updating gene {gene} -> {csv_path}")
                        
                        logger.info(f"Final results to save: {list(existing_results.keys())}")
                        
                        await prisma.trainingrun.update(
                            where={"id": experiment_id},
                            data={"geneDepmapResults": Json(existing_results)}
                        )
                        logger.info(f"✓✓✓ Successfully updated database with DepMap results for experiment {experiment_id}")
                    except Exception as e:
                        logger.error(f"Database update error: {e}")
                        import traceback
                        logger.error(traceback.format_exc())
                
                # Execute async function (create new event loop)
                try:
                    # Try to get existing loop
                    try:
                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            # If loop is running, we need to use a different approach
                            import concurrent.futures
                            with concurrent.futures.ThreadPoolExecutor() as executor:
                                future = executor.submit(asyncio.run, update_database())
                                future.result(timeout=10)
                        else:
                            loop.run_until_complete(update_database())
                    except RuntimeError:
                        # No event loop, create new one
                        asyncio.run(update_database())
                except Exception as e:
                    logger.error(f"Failed to run database update: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    
            except Exception as db_error:
                # Don't fail the job if database update fails
                logger.error(f"Failed to update database with DepMap results: {db_error}")
                import traceback
                logger.error(traceback.format_exc())
        
        return {
            "csv_path": str(csv_path),
            "gene_count": len(genes),
            "association_count": len(combined),
        }
        
    except Exception as e:
        logger.error(f"Error processing DepMap associations: {e}")
        logger.error(traceback.format_exc())
        raise

