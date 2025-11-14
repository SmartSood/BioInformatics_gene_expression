from fastapi import APIRouter, UploadFile, File, Depends
from auth.deps import get_current_user
from client.db import db
router = APIRouter(prefix="/dataset", tags=["dataset"])

@router.get("")
async def read_datasets( user=Depends(get_current_user)):
    try:
        datasets = await db.dataset.find_many(
            where={"userId": int(user["sub"])}
        )
        return {"datasets": datasets}
    except Exception as e:
        print(f"Error fetching datasets for user {user['sub']}: {e}")
        return {"datasets": []}
    