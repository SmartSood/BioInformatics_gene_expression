# Embedding Backend

FastAPI microservice for generating drug+gene embeddings from `apps/embedding_bundle` with:
- Hybrid API (`/embeddings/sync` + `/embeddings/async`)
- RQ worker queue (`embedding`)
- Cached model loading per process (no repeated HF downloads)
- JSON response plus optional ZIP artifact downloads

## Run locally

```bash
cd /Users/smarthsood/Desktop/Gene_startup/gene_web
source apps/model_backend/venv/bin/activate
export REDIS_URL="${REDIS_URL:-redis://localhost:6379/0}"
python -m uvicorn apps.embedding_backend.server:app --reload --port 8002
```

Run worker:

```bash
cd /Users/smarthsood/Desktop/Gene_startup/gene_web
source apps/model_backend/venv/bin/activate
export REDIS_URL="${REDIS_URL:-redis://localhost:6379/0}"
python -m apps.embedding_backend.workers.run_rq_worker
```

## Required env vars

- `JWT_SECRET`
- `AUTH_JWT_ISSUER`
- `AUTH_JWT_AUDIENCE`
- `REDIS_URL`

Optional:
- `EMBEDDING_MODELS_DIR`
- `EMBEDDING_HF_CACHE_DIR`
- `EMBEDDING_OUTPUT_DIR`
- `EMBEDDING_DEVICE` (`cpu` or `cuda`)

## Endpoints

- `POST /embeddings/sync`
- `POST /embeddings/async`
- `GET /embeddings/{job_id}/status`
- `GET /embeddings/{job_id}/download?format=zip|metadata|drug|gene|combined`
- `GET /health`

