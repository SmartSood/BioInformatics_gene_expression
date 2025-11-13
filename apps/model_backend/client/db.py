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
            await db.connect()
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
