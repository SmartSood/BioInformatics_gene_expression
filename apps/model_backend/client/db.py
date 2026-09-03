# client/db.py
import sys
import os
import atexit
import asyncio
from typing import Optional

PRISMA_PARENT = "/Users/smarthsood/Desktop/Gene_startup/gene_web/packages/db/generated/python"
if PRISMA_PARENT not in sys.path:
    sys.path.insert(0, PRISMA_PARENT)

from prisma import Prisma

db = Prisma()

async def connect_db() -> Prisma:
    """Ensure prisma client is connected in this process/event loop."""
    try:
        if not db.is_connected():
            # Default engine-connect timeout (~10s) is too tight for this
            # binary's cold-start time on this hardware - startup logs
            # showed the connection attempt failing right around 10s in.
            await db.connect(timeout=30)
    except Exception:
        # bubble up or let caller handle (we'll print from caller)
        raise
    return db

async def disconnect_db() -> None:
    try:
        if db.is_connected():
            await db.disconnect()
    except Exception:
        # best-effort
        pass

def _atexit_disconnect():
    # called synchronously on process exit; run event loop to disconnect
    try:
        asyncio.run(disconnect_db())
    except Exception:
        pass

atexit.register(_atexit_disconnect)
