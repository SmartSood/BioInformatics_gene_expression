from fastapi import APIRouter

from apps.embedding_backend.services.embedding_service import ensure_required_assets

router = APIRouter(prefix="/health", tags=["health"])


@router.get("")
async def health():
    try:
        ensure_required_assets()
        return {"ok": True, "models": "ready"}
    except Exception as exc:  # pragma: no cover
        return {"ok": False, "models": "missing", "error": str(exc)}

