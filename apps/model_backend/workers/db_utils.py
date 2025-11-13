# workers/db_utils.py

import time
import traceback
import math
import json
import asyncio
import logging
from typing import Any, Optional
from datetime import datetime, date
from decimal import Decimal

# If numpy / pandas exist in your env, import them lazily
try:
    import numpy as np
except Exception:
    np = None

try:
    import pandas as pd
except Exception:
    pd = None

# Import Prisma Json helper for proper JSON field wrapping
from prisma import Json

# logger (worker already configures a handler; this will reuse it)
logger = logging.getLogger("train_worker.db_utils")
logger.setLevel(logging.INFO)


def _to_json_serializable(obj: Any):
    """
    Recursively convert common non-JSON-safe types to plain Python types.
    Returns a JSON-safe structure (dict/list/primitives) or raises TypeError if impossible.
    """

    # None / bool / str
    if obj is None or isinstance(obj, (bool, str)):
        return obj

    # ints
    if isinstance(obj, int) and not isinstance(obj, bool):
        return obj

    # Decimal -> float (or str fallback)
    if isinstance(obj, Decimal):
        try:
            f = float(obj)
            if not math.isfinite(f):
                return None
            return f
        except Exception:
            return str(obj)

    # datetime/date -> ISO
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()

    # floats: NaN/inf -> None
    if isinstance(obj, float):
        if not math.isfinite(obj):
            return None
        return obj

    # numpy scalar -> python scalar
    if np is not None and isinstance(obj, np.generic):
        return _to_json_serializable(obj.item())

    # numpy array -> list (recurse)
    if np is not None and isinstance(obj, np.ndarray):
        return _to_json_serializable(obj.tolist())

    # pandas types
    if pd is not None:
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        if obj is pd.NA:
            return None
        if isinstance(obj, (pd.Series, pd.Index)):
            return _to_json_serializable(obj.tolist())
        if isinstance(obj, pd.DataFrame):
            # convert to list-of-dicts (records)
            return _to_json_serializable(obj.to_dict(orient="records"))

    # dict -> sanitize keys and values
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            # JSON object keys must be strings
            if not isinstance(k, str):
                k_str = str(k)
            else:
                k_str = k
            out[k_str] = _to_json_serializable(v)
        return out

    # list/tuple/set -> list
    if isinstance(obj, (list, tuple, set)):
        return [_to_json_serializable(v) for v in obj]

    # try to use tolist/to_dict where available (fallback)
    try:
        if hasattr(obj, "tolist"):
            return _to_json_serializable(obj.tolist())
    except Exception:
        pass

    try:
        if hasattr(obj, "to_dict"):
            return _to_json_serializable(obj.to_dict())
    except Exception:
        pass

    # last-resort: try to JSON-dump directly
    try:
        json.dumps(obj)
        return obj
    except Exception:
        # fallback to str() so Prisma receives something predictable
        try:
            return str(obj)
        except Exception:
            raise TypeError(f"Cannot convert object of type {type(obj)} to JSON-safe value")


def sanitize_metrics(metrics: Any) -> Optional[Any]:
    """
    Convert `metrics` to a JSON-serializable Python structure suitable for Prisma Json.
    Returns a JSON-safe object (dict/list/primitive) or None.
    """
    if metrics is None:
        return None

    try:
        sanitized = _to_json_serializable(metrics)

        # Final validation: ensure json.dumps succeeds (Prisma expects JSON-compatible)
        try:
            json.dumps(sanitized)
        except TypeError:
            # Try converting some problematic values to strings and try again
            def force_str(v):
                if isinstance(v, dict):
                    return {k: force_str(val) for k, val in v.items()}
                if isinstance(v, list):
                    return [force_str(x) for x in v]
                try:
                    json.dumps(v)
                    return v
                except Exception:
                    return str(v)

            sanitized = force_str(sanitized)
            # If still failing, last fallback is None
            try:
                json.dumps(sanitized)
            except Exception:
                logger.warning("sanitize_metrics: final payload not JSON-dumpable after coercion; returning None")
                return None

        return sanitized
    except Exception:
        logger.exception("sanitize_metrics: unexpected error while sanitizing metrics; returning safe string")
        try:
            return str(metrics)
        except Exception:
            return None


async def update_trainingrun_with_retries(prisma_client, job_id: str, data: dict, attempts: int = 3, base_delay: float = 0.5):
    """
    Try prisma_client.trainingrun.update(...) with retries and full logging on failure.
    This is async and will await the prisma client's coroutine. Raises the last exception if all attempts fail.
    """
    last_exc = None

    # Defensive: ensure metrics (if present) is sanitized here to avoid schema parse errors
    if "metrics" in data:
        try:
            data["metrics"] = sanitize_metrics(data["metrics"])
        except Exception:
            logger.exception("Failed to sanitize metrics for job %s; setting metrics=None", job_id)
            data["metrics"] = None

    # ✅ Wrap metrics in Prisma.Json for proper DB update
    if "metrics" in data and data["metrics"] is not None:
        data["metrics"] = Json(data["metrics"])

    # Log the payload we are going to send (truncate large payloads)
    try:
        payload_json = json.dumps(data, default=str)
    except Exception:
        payload_json = repr(data)
    logger.info("Prisma update attempt for job %s; payload (truncated): %s", job_id, payload_json[:2000])

    for i in range(1, attempts + 1):
        try:
            # Prisma client update is async; await it
            result = await prisma_client.trainingrun.update(where={"id": job_id}, data=data)
            logger.info("Prisma update successful for job %s on attempt %d", job_id, i)
            return result
        except Exception as e:
            last_exc = e
            logger.warning("Warning: update attempt %d/%d failed for job %s: %s", i, attempts, job_id, e)
            logger.warning(traceback.format_exc())
            if i < attempts:
                # exponential backoff (async)
                sleep_for = min(5, base_delay * (2 ** (i - 1)))
                logger.info("Sleeping %.2fs before next update attempt for job %s", sleep_for, job_id)
                await asyncio.sleep(sleep_for)

    logger.error("Final update attempt failed for job %s after %d attempts", job_id, attempts)
    # Raise the last exception to let caller handle it (and so RQ marks job failed)
    raise last_exc


# Optional small test harness you can run: `python workers/db_utils.py`
if __name__ == "__main__":
    # quick local check
    test_metrics = {
        "acc": 0.9234,
        "loss": float("nan"),
        "confusion": np.array([[1, 2], [3, 4]]) if np is not None else [[1, 2], [3, 4]],
        "created": datetime.utcnow(),
        "extra": Decimal("0.1"),
        ("tuple", 1): "key-was-not-str",  # non-string key
    }
    print("raw:", test_metrics)
    print("sanitized:", sanitize_metrics(test_metrics))
    try:
        print("json ok:", json.dumps(sanitize_metrics(test_metrics), default=str)[:2000])
    except Exception as exc:
        print("json dump failed:", exc)
