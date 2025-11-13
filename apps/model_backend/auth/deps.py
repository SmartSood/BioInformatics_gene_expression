from __future__ import annotations
import os
from typing import Annotated, Optional, Callable, Iterable
from fastapi import Depends, Header, HTTPException, status

# Load .env in dev
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

AUD = os.getenv("AUTH_JWT_AUDIENCE")
ISS = os.getenv("AUTH_JWT_ISSUER")
SECRET = os.getenv("JWT_SECRET")
LEEWAY_SECONDS = int(os.getenv("AUTH_JWT_LEEWAY", "30"))

# Postman usage tips:
# - Authorization header: Bearer <JWT>
# - File upload field name must match the endpoint parameter (e.g. `file` if signature is file: UploadFile = File(...))
# - For multiple files use the declared parameter name (e.g. `files`)

def _config_ok():
    missing = []
    if not SECRET: missing.append("JWT_SECRET")
    if not ISS: missing.append("AUTH_JWT_ISSUER")
    if not AUD: missing.append("AUTH_JWT_AUDIENCE")
    if missing:
        raise HTTPException(500, detail=f"Missing server configuration: {', '.join(missing)}")

def _extract_bearer(authorization: Optional[str]) -> str:
    if not authorization:
        raise HTTPException(401, "Missing Authorization header")
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(401, "Invalid Authorization header")
    return token

def _decode_hs256(token: str) -> dict:
    _config_ok()
    # First try PyJWT (PyPI: PyJWT). If the installed 'jwt' is the wrong package, fall back to python-jose.
    # PyJWT path
    try:
        import jwt as pyjwt  # type: ignore
        decode = getattr(pyjwt, "decode", None)
        if not callable(decode):
            raise AttributeError("jwt.decode not found (likely wrong 'jwt' package installed)")
        return decode(
            token,
            SECRET,
            algorithms=["HS256"],
            audience=AUD,
            issuer=ISS,
            options={
                "require": ["exp", "iss", "aud", "sub"],
                "verify_signature": True,
                "verify_exp": True,
                "verify_iss": True,
                "verify_aud": True,
                "verify_sub": False,  # allow non-string sub, we'll normalize after decode
            },
            leeway=LEEWAY_SECONDS,
        )
    except Exception as e_pyjwt:
        # Map common PyJWT exceptions if they exist; otherwise, try python-jose.
        try:
            from jwt import exceptions as jwt_exceptions  # type: ignore
        except Exception:
            jwt_exceptions = None

        def _is(err, name: str) -> bool:
            if err.__class__.__name__ == name:
                return True
            if jwt_exceptions is not None:
                t = getattr(jwt_exceptions, name, None)
                if t is not None and isinstance(err, t):
                    return True
            return False

        if _is(e_pyjwt, "ExpiredSignatureError"):
            raise HTTPException(401, "Token expired")
        if _is(e_pyjwt, "InvalidAudienceError"):
            raise HTTPException(401, "Invalid audience")
        if _is(e_pyjwt, "InvalidIssuerError"):
            raise HTTPException(401, "Invalid issuer")

        # If it looks like a wrong/unsupported jwt library, try python-jose next.
        if isinstance(e_pyjwt, (AttributeError, ImportError)):
            pass
        else:
            # Any other PyJWT error -> invalid token
            raise HTTPException(401, f"Invalid token: {e_pyjwt}")

        # python-jose fallback
        try:
            from jose import jwt as jose_jwt  # type: ignore
        except Exception:
            raise HTTPException(500, "No compatible JWT library found. Install 'PyJWT' or 'python-jose'.")

        try:
            return jose_jwt.decode(
                token,
                SECRET,
                algorithms=["HS256"],
                audience=AUD,
                issuer=ISS,
                options={
                    "verify_signature": True,
                    "verify_exp": True,
                    "verify_aud": True,
                    "verify_iat": False,
                    "require_exp": True,
                },
                leeway=LEEWAY_SECONDS,
            )
        except jose_jwt.ExpiredSignatureError:
            raise HTTPException(401, "Token expired")
        except jose_jwt.JWTClaimsError as e:
            msg = str(e).lower()
            if "audience" in msg:
                raise HTTPException(401, "Invalid audience")
            if "issuer" in msg:
                raise HTTPException(401, "Invalid issuer")
            raise HTTPException(401, f"Invalid claims: {e}")
        except jose_jwt.JWTError as e:
            raise HTTPException(401, f"Invalid token: {e}")

async def get_current_user(authorization: Annotated[Optional[str], Header()] = None):
    token = _extract_bearer(authorization)
    payload = _decode_hs256(token)

    scopes = payload.get("scope") or payload.get("scopes") or []
    if isinstance(scopes, str):
        scopes = [s for s in scopes.split() if s]

    sub = payload.get("sub")
    if sub is None or (isinstance(sub, str) and not sub.strip()):
        raise HTTPException(401, "Invalid token: missing 'sub'")

    user = {"sub": str(sub), "email": payload.get("email"), "scope": scopes}
    return user

def require_scopes(required: Iterable[str]) -> Callable:
    required_set = set(required)
    async def _checker(user=Depends(get_current_user)):
        user_scopes = set(user.get("scope") or [])
        if not required_set.issubset(user_scopes):
            missing = sorted(required_set - user_scopes)
            raise HTTPException(403, f"Missing required scope(s): {missing}")
        return True
    return _checker
