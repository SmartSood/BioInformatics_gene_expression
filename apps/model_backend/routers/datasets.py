from fastapi import APIRouter, UploadFile, File, Depends
from auth.deps import get_current_user
from storage.storage import save_upload
from scemma.model import DatasetInfoRequest
router = APIRouter(prefix="/datasets", tags=["datasets"])

@router.post("")
async def upload_dataset(
    req: DatasetInfoRequest = Depends(DatasetInfoRequest.as_form),
    file: UploadFile = File(...),
    user=Depends(get_current_user)
):
    ds_id, uri = await save_upload(req.name, req.description, file, owner_id=user["sub"])
    return {"dataset_id": ds_id, "uri": uri}