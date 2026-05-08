# DepMap Backend

A separate microservice for processing gene-drug interaction associations using DepMap datasets.

## Setup

1. **Install dependencies** (same as model_backend):
   ```bash
   pip install fastapi uvicorn redis rq pandas numpy python-dotenv PyJWT
   ```

2. **Environment variables** (same as model_backend):
   - `JWT_SECRET` - JWT secret key
   - `AUTH_JWT_ISSUER` - JWT issuer
   - `AUTH_JWT_AUDIENCE` - JWT audience
   - `REDIS_URL` - Redis connection URL (default: `redis://localhost:6379/0`)

## Running

### Start the FastAPI server:
```bash
cd /Users/smarthsood/Desktop/Gene_startup/gene_web
python -m uvicorn apps.depmap_backend.server:app --reload --port 8001
```

### Start the RQ worker (in a separate terminal):
```bash
cd /Users/smarthsood/Desktop/Gene_startup/gene_web
source apps/model_backend/venv/bin/activate  # or your virtual environment
export REDIS_URL="${REDIS_URL:-redis://localhost:6379/0}"
python -m apps.depmap_backend.workers.run_rq_worker
```

**macOS:** Use the command above. It runs RQ’s **`SimpleWorker`** on Darwin (no `fork()` per job), which avoids the `objc ... fork() was called` crash when pandas/numpy are loaded. Plain **`rq worker ...`** forks each job; **`OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES`** helps some setups but is **not reliable** on many newer Macs, which is why DepMap uses `SimpleWorker` here.

**Linux:** The same module uses the standard RQ `Worker` (forking is fine).

**Note**: The worker uses the same Redis instance as model_backend but a separate queue name (`depmap`) to keep jobs isolated.

## API Endpoints

- `POST /associations` - Create a new gene association analysis job
- `GET /associations/{job_id}/status` - Get job status
- `GET /associations/{job_id}/download` - Download CSV results

## Architecture

- **Separate microservice**: Runs on port 8001 (different from model_backend on 8000)
- **Shared Redis**: Uses same Redis instance but separate queue (`depmap` vs `train`)
- **Dataset caching**: Datasets are loaded once per worker process and cached
- **Output storage**: CSV files stored in `apps/depmap_backend/outputs/`

## Frontend Integration

The frontend has:
- "DepMap" button on each gene card in experiment details
- DepMap results page at `/dashboard/depmap?gene=GENE_SYMBOL`
- Real-time polling for job status
- CSV download functionality

