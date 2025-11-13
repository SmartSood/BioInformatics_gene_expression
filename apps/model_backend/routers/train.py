from fastapi import APIRouter, Depends, HTTPException
from auth.deps import get_current_user
from apps.model_backend.workers.queue_worker import get_queue
from scemma.model import TrainRequest, TrainResponse
from client.db import db
from workers.train_worker import run_train
router = APIRouter(prefix="/train", tags=["train"])

@router.post("", response_model=TrainResponse)
async def start_training(req: TrainRequest, user=Depends(get_current_user)):
    if not user["sub"]:
        raise HTTPException(401, "No subject in token")

    q = get_queue()
    job = q.enqueue(
    run_train,                                # ✅ pass function object
    req.dataset_uri,
    req.config.model_dump(),
    user["sub"],
    job_timeout=60 * 60,
)

    # Record the run immediately
    await db.trainingrun.upsert(
        where={"id": job.id},
        data={
            "create": {
                "id": job.id,
                "userId": int(user["sub"]),          # adjust if sub isn’t an int
                "status": "queued",
                "datasetUri": req.dataset_uri,
            },
            "update": {
                "status": "queued",
                "datasetUri": req.dataset_uri,
            },
        },
    )

    return {"job_id": job.id}
