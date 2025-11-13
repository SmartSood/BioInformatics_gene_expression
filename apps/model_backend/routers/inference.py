from fastapi import APIRouter, Depends, HTTPException
from joblib import load
from pathlib import Path
import os
import pandas as pd

from auth.deps import get_current_user
from scemma.model import PredictRequest, PredictResponse

router = APIRouter(prefix="/models", tags=["inference"])

ARTIFACTS_DIR = os.getenv("ARTIFACTS_DIR", "./artifacts")
MODEL_FILENAME = "model.joblib"

def _model_path(user_id: str, job_id: str) -> Path:
    path = Path(ARTIFACTS_DIR) / str(user_id) / job_id / MODEL_FILENAME
    return path

@router.post("/{job_id}/predict", response_model=PredictResponse)
async def predict_by_job(job_id: str, req: PredictRequest, user=Depends(get_current_user)):
    path = _model_path(user["sub"], job_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Model not found for this job")

    pipe = load(path)
    df = pd.DataFrame(req.records)

    preds = pipe.predict(df)
    proba = getattr(pipe, "predict_proba", None)
    probs = proba(df).tolist() if callable(proba) else None

    return {"predictions": preds.tolist(), "probabilities": probs}
