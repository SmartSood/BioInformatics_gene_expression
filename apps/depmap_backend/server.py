# apps/depmap_backend/server.py
"""
Entry point for the DepMap FastAPI app.
This microservice handles gene-drug interaction analysis using DepMap datasets.
"""

# --- Path hack: make project root and app dir available on sys.path ---
import sys
from pathlib import Path

# PROJECT_ROOT = <project>/ (one level above 'apps')
# APP_DIR = <project>/apps/depmap_backend
PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_DIR = Path(__file__).resolve().parent

# Insert project root and app dir at front of sys.path if not present.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

# --- Standard app imports (after sys.path is set) ---
import logging
import os
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

# Import routers
from apps.depmap_backend.routers import associations, health

# --- logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger("apps.depmap_backend.server")

origins = [
    o.strip()
    for o in os.getenv(
        "CORS_ALLOWED_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000"
    ).split(",")
    if o.strip()
]

# --- create app and include routers ---
def create_app() -> FastAPI:
    app = FastAPI(title="DepMap API")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router)
    app.include_router(associations.router)
    return app


app = create_app()

