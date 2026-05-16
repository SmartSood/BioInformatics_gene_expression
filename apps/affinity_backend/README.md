# Affinity Backend

FastAPI microservice for affinity prediction from uploaded embeddings CSV using
the model/checkpoint in `apps/affinity`.

## Run locally

```bash
cd /Users/smarthsood/Desktop/Gene_startup/gene_web
source apps/model_backend/venv/bin/activate
python -m uvicorn apps.affinity_backend.server:app --reload --port 8003
```

## Required env vars

- `JWT_SECRET`
- `AUTH_JWT_ISSUER`
- `AUTH_JWT_AUDIENCE`

## Optional env vars

- `AFFINITY_CHECKPOINT_PATH` (default: `apps/affinity/gene_embeddings.pth`)
- `AFFINITY_OUTPUT_DIR` (default: `apps/affinity_backend/outputs`)

## Endpoints

- `GET /health`
- `POST /affinity/predict` (multipart file upload, field name: `file`)
- `GET /affinity/{request_id}/download`
- `GET /affinity/sample-csv`
