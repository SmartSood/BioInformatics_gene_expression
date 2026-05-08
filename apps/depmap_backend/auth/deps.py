"""
Authentication dependencies for DepMap backend.
Reuses the same JWT validation as model_backend.
"""
from __future__ import annotations
from typing import Annotated, Optional, Dict
from fastapi import Header

# Import directly from model_backend using absolute import
# This avoids circular import issues
from apps.model_backend.auth.deps import get_current_user as _get_current_user_base

# Re-export the same function
async def get_current_user(authorization: Annotated[Optional[str], Header()] = None) -> Dict:
    """
    Validate JWT token and return user info.
    Reuses the same auth logic as model_backend.
    """
    return await _get_current_user_base(authorization)

