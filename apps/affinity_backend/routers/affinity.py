from __future__ import annotations

import io
from typing import Any, Dict

import pandas as pd
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import Response

from apps.affinity_backend.auth.deps import get_current_user
from apps.affinity_backend.services.affinity_service import (
    build_sample_csv,
    predict_affinity,
    required_feature_prefixes,
)

router = APIRouter(prefix="/affinity", tags=["affinity"])


def _require_user_id(user: Dict[str, Any]) -> str:
    sub = user.get("sub")
    if not sub:
        raise HTTPException(401, "No subject in token")
    return str(sub)


@router.post("/predict")
async def predict_affinity_from_csv(
    file: UploadFile = File(...),
    user: Dict[str, Any] = Depends(get_current_user),
):
    user_id = _require_user_id(user)

    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(400, "Please upload a .csv file")

    try:
        raw = await file.read()
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as exc:
        raise HTTPException(400, f"Failed to parse CSV: {exc}") from exc

    if df.empty:
        raise HTTPException(400, "Uploaded CSV is empty")

    try:
        out_df = predict_affinity(df)
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc

    predicted_affinity = float(out_df.iloc[0]["predicted_affinity"])

    return {
        "predicted_affinity": predicted_affinity,
        "row_count": int(len(out_df)),
        "required_columns": required_feature_prefixes(),
    }


@router.get("/sample-csv")
async def download_sample_csv(
    user: Dict[str, Any] = Depends(get_current_user),
):
    _ = _require_user_id(user)
    df = build_sample_csv()
    content = df.to_csv(index=False)
    return Response(
        content=content,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=affinity_input_sample.csv"},
    )
