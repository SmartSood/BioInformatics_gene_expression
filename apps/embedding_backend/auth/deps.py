from __future__ import annotations

from typing import Annotated, Dict, Optional

from fastapi import Header

from apps.model_backend.auth.deps import get_current_user as _get_current_user_base


async def get_current_user(
    authorization: Annotated[Optional[str], Header()] = None,
) -> Dict:
    return await _get_current_user_base(authorization)

