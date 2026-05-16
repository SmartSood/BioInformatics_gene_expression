import logging
import sys
from pathlib import Path

from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_DIR = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from apps.affinity_backend.routers import affinity, health

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger("apps.affinity_backend.server")

origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]


def create_app() -> FastAPI:
    app = FastAPI(title="Affinity Backend API")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(health.router)
    app.include_router(affinity.router)
    return app


app = create_app()
