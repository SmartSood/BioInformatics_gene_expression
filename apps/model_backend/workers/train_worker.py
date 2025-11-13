# workers/train_worker.py
import os
from pathlib import Path
import asyncio
from typing import Optional
from rq import get_current_job
import logging

logger = logging.getLogger("worker")
logger.setLevel(logging.INFO)

fh = logging.FileHandler("worker_debug.log")
fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
logger.addHandler(fh)


# lightweight DB helper (should be safe to import)
from client.db import db, connect_db

# db_utils does lazy imports for numpy/pandas, so safe to import at module level
from workers.db_utils import sanitize_metrics, update_trainingrun_with_retries

ARTIFACTS_DIR = os.getenv("ARTIFACTS_DIR", "./artifacts")


def run_train(dataset_uri: str, config: dict, owner_id: str):
    """
    Worker entrypoint for training jobs.
    - Defers himports (pipeline) until runtime so native libs initialize in worker child.
    - Connects to Prisma (if available) and marks started/finished/failed states.
    - Sanitizes metrics before sending to Prisma and retries updates on transient failures.
    """
    job = get_current_job()
    job_id: Optional[str] = job.id if job else "no_jobid"
    job_id = str(job_id)

    out_dir = Path(ARTIFACTS_DIR) / str(owner_id) / job_id
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = dataset_uri
    model_path = str(out_dir / "model.joblib")

    # Import heavy pipeline here so numpy/pandas/OpenBLAS etc. initialize inside the child process.
    # This reduces fork-related native crashes.
    from pipeline.pipeline import train  # moved inside function

    async def _run():
        prisma = None
        try:
            # Ensure DB is connected *in this process / event loop* (idempotent)
            try:
                prisma = await connect_db()
            except Exception as e:
                prisma = None
                logger.warning("Warning: failed to connect to DB at job start: %s", e)

            # Upsert -> mark started (safe if prisma is None)
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

            # Do the heavy-lifting training (sync function)
            result = train(dataset_path, config, str(out_dir))
            metrics = (result or {}).get("metrics")

            # Sanitize metrics to JSON-safe types before sending to Prisma
            metrics_clean = sanitize_metrics(metrics)

            # Attempt to update final status with retries and verbose logging on failure
            if prisma:
                try:
                    await update_trainingrun_with_retries(
                        prisma,
                        job_id,
                        {
                            "status": "finished",
                            "modelPath": model_path,
                            "metrics": metrics_clean,
                        },
                        attempts=3,
                        base_delay=0.5,
                    )
                    logger.info(f"DB updated to finished for job {job_id}")
                except Exception:
                    logger.critical("final DB update failed for job %s — see traceback below:", job_id, exc_info=True)
            # Return enriched result for RQ
            enriched = dict(result or {})
            enriched.setdefault("job_id", job_id)
            enriched.setdefault("model_path", model_path)
            return enriched

        except Exception as e:
            # Mark failed in DB (best-effort)
            if prisma:
                try:
                    try:
                        await update_trainingrun_with_retries(
                            prisma,
                            job_id,
                            {"status": "failed"},
                            attempts=2,
                            base_delay=0.2,
                        )
                    except Exception:
                        # If update fails, log but continue to raise original exception
                        logger.warning("marking job failed in DB also failed for job %s", job_id, exc_info=True)
                except Exception:
                    pass
            # Re-raise so RQ knows the job errored
            raise

        # NOTE: DO NOT disconnect here — the shared db instance should remain connected
        # until the process exits. atexit handler or worker shutdown will disconnect.

    return asyncio.run(_run())
