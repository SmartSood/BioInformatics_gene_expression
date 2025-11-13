# apps/model_backend/server.py
"""
Entry point for the FastAPI app. This file:
 - ensures imports work both as package-qualified (apps.*) and bare local imports (routers)
 - registers startup/shutdown handlers to connect/disconnect the Prisma client once per process
 - mounts routers
"""

# --- Path hack: make project root and app dir available on sys.path so existing imports work ---
import sys
from pathlib import Path

# PROJECT_ROOT = <project>/ (one level above 'apps')
# APP_DIR = <project>/apps/model_backend
PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_DIR = Path(__file__).resolve().parent

# Insert project root and app dir at front of sys.path if not present.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))


# --- Standard app imports (after sys.path is set) ---
import logging
import asyncio
from fastapi import FastAPI

# Use package-qualified imports to be explicit where possible; bare imports in modules
# will still work because APP_DIR is on sys.path
from apps.model_backend.routers import health, datasets, train, jobs, inference

# Import the DB helpers (client.db must exist under apps.model_backend/client/db.py)
# connect_db and disconnect_db should be async functions
from client.db import connect_db, disconnect_db

# --- logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger("apps.model_backend.server")

# --- create app and include routers ---
def create_app() -> FastAPI:
    app = FastAPI(title="ML API")

    app.include_router(health.router)
    app.include_router(datasets.router)
    app.include_router(train.router)
    app.include_router(jobs.router)
    app.include_router(inference.router)

    return app


app = create_app()


# --- startup / shutdown lifecycle events for DB connection management ---

@app.on_event("startup")
async def on_startup_connect_db():
    """
    Ensure the Prisma client is connected once when the process starts.
    This avoids per-request connect/disconnect overhead and prevents 'ClientNotConnectedError'.
    """
    try:
        logger.info("Startup: connecting to Prisma DB...")
        # connect_db() is expected to be idempotent; it returns quickly if already connected.
        await connect_db()
        logger.info("Startup: Prisma DB connected.")
    except Exception as e:
        # Log exception in detail. We intentionally don't re-raise here so the app can start
        # and handle DB errors on a per-request basis. If you prefer failing fast, re-raise.
        logger.exception("Startup: failed to connect Prisma DB: %s", e)


@app.on_event("shutdown")
async def on_shutdown_disconnect_db():
    """
    Disconnect the Prisma client when the process exits cleanly.
    """
    try:
        logger.info("Shutdown: disconnecting Prisma DB...")
        await disconnect_db()
        logger.info("Shutdown: Prisma DB disconnected.")
    except Exception as e:
        logger.exception("Shutdown: failed to disconnect Prisma DB: %s", e)
