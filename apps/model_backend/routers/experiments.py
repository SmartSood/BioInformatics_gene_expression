from fastapi import APIRouter, Depends, HTTPException
from auth.deps import get_current_user
from client.db import db
router = APIRouter(prefix="/experiments", tags=["experiments"])

@router.get("")
async def list_experiments(user=Depends(get_current_user)):
    if not user["sub"]:
        raise HTTPException(401, "No subject in token")

    # Placeholder: Fetch experiments from the database
    experiments = await db.trainingrun.find_many(
        where={"userId": int(user["sub"])}
    )

    return {"experiments": experiments}