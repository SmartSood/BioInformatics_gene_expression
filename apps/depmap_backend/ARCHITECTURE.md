# DepMap Backend Architecture

## How DepMap Backend Works: From Script to Microservice

### Original Script (`depmap_associations.py`)

The original script is a command-line tool:
```bash
python3 depmap_associations.py --genes ERCC3 --output ERCC3_associations.csv
```

It:
1. Takes gene names as command-line arguments
2. Loads datasets (expression, GDSC, CTRP, PRISM)
3. Computes correlations between gene expression and drug sensitivity
4. Writes results to a CSV file

---

## Architecture Transformation

### Step 1: Extract Reusable Functions

The script has internal functions that do the actual work:
- `_load_expression()` - Loads gene expression data
- `_load_gdsc_auc()` - Loads GDSC drug sensitivity data
- `_load_ctrp_auc()` - Loads CTRP drug sensitivity data
- `_load_prism_matrix()` - Loads PRISM data
- `_compute_correlations()` - Computes Pearson correlations

These functions are imported and reused in the worker.

---

### Step 2: Create FastAPI Microservice Structure

```
apps/depmap_backend/
├── server.py              # FastAPI app entry point
├── routers/
│   ├── associations.py    # API endpoints
│   └── health.py         # Health check
├── workers/
│   ├── depmap_worker.py  # Background job processor
│   └── queue_worker.py   # Redis queue setup
└── auth/
    └── deps.py           # JWT authentication
```

---

### Step 3: API Layer (`routers/associations.py`)

Converts HTTP requests into background jobs:

```python
@router.post("/associations")
async def create_gene_association(req: GeneAssociationRequest, user=...):
    # 1. Validate request (genes, user auth)
    # 2. Create Redis job queue
    # 3. Enqueue the worker function
    job = q.enqueue(run_depmap_association, genes, user_id, experiment_id)
    # 4. Return job_id immediately (async processing)
    return {"job_id": job.id, "status": "queued"}
```

**Why async?**
- Analysis can take minutes
- Returns immediately with a job ID
- Client polls for status

---

### Step 4: Worker Function (`workers/depmap_worker.py`)

This is where the original script logic runs:

```python
def run_depmap_association(genes, user_id, experiment_id):
    # 1. Import functions from original script
    from depmap_associations import (
        _load_expression,
        _load_gdsc_auc,
        _compute_correlations,
        ...
    )
    
    # 2. Load datasets (with caching!)
    expression = _load_expression(expression_path, genes)
    model = _load_cached_model()  # Cached per worker process
    
    # 3. Process each dataset (GDSC, CTRP, PRISM)
    gdsc_res = _compute_correlations(expression, gdsc, ...)
    ctrp_res = _compute_correlations(expression, ctrp, ...)
    ...
    
    # 4. Combine results
    combined = pd.concat([gdsc_res, ctrp_res, ...])
    
    # 5. Save to organized directory
    output_dir = f"outputs/{user_id}/{experiment_id}/"
    csv_path = output_dir / f"{job_id}_associations.csv"
    combined.to_csv(csv_path)
    
    return {"csv_path": str(csv_path), ...}
```

**Improvements over the script:**
1. **Dataset caching**: Loads datasets once per worker process (not per job)
2. **Organized storage**: `outputs/{userId}/{experimentId}/`
3. **Error handling**: Logs errors, returns structured results
4. **Background processing**: Runs in separate worker process

---

### Step 5: Redis Job Queue

Uses Redis Queue (RQ) for background processing:

```python
# queue_worker.py
def get_queue():
    return Queue("depmap", connection=Redis.from_url(REDIS_URL))
```

**Flow:**
1. API receives request → creates job → returns job_id
2. RQ worker picks up job → runs `run_depmap_association()`
3. Worker processes → saves CSV → returns result
4. Client polls `/associations/{job_id}/status` → gets status
5. When finished → client downloads CSV

---

## Complete Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│ 1. USER CLICKS "DepMap" BUTTON                             │
├─────────────────────────────────────────────────────────────┤
│ Frontend: ExperimentDetails.tsx                            │
│   - Gets gene symbol (e.g., "ERCC3")                       │
│   - Gets experiment ID                                      │
│   - Navigates to /dashboard/depmap?gene=ERCC3&expId=...    │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. FRONTEND SENDS API REQUEST                               │
├─────────────────────────────────────────────────────────────┤
│ depmap/page.tsx:                                            │
│   POST /associations                                        │
│   Body: { genes: ["ERCC3"], experiment_id: "abc123" }      │
│   Headers: Authorization: Bearer <token>                   │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. API ENDPOINT (associations.py)                          │
├─────────────────────────────────────────────────────────────┤
│   - Validates JWT token                                     │
│   - Validates request (genes, experiment_id)                │
│   - Creates Redis job:                                      │
│     q.enqueue(run_depmap_association, genes, userId, expId)│
│   - Returns: { job_id: "xyz", status: "queued" }          │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. REDIS QUEUE                                              │
├─────────────────────────────────────────────────────────────┤
│   - Job stored in Redis queue "depmap"                     │
│   - Status: "queued" → "started" → "finished"             │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 5. RQ WORKER PROCESSES JOB                                 │
├─────────────────────────────────────────────────────────────┤
│ Worker: depmap_worker.py                                   │
│                                                              │
│   a) Import functions from depmap_associations.py:          │
│      - _load_expression()                                   │
│      - _load_gdsc_auc()                                     │
│      - _compute_correlations()                              │
│                                                              │
│   b) Load datasets (with caching):                         │
│      - Expression: Load gene expression for "ERCC3"        │
│      - Model: Load cell line metadata (cached)             │
│      - GDSC: Load drug sensitivity (cached if exists)     │
│      - CTRP: Load drug sensitivity (cached if exists)      │
│      - PRISM: Load drug sensitivity (cached if exists)    │
│                                                              │
│   c) Compute correlations:                                 │
│      For each dataset:                                      │
│        correlation = _compute_correlations(                 │
│          expression, drug_matrix, dataset_label            │
│        )                                                    │
│                                                              │
│   d) Combine results:                                       │
│      combined = pd.concat([gdsc_res, ctrp_res, ...])       │
│      Sort by absolute correlation (descending)             │
│                                                              │
│   e) Save CSV:                                              │
│      Path: outputs/11/abc123/job-xyz_associations.csv      │
│      combined.to_csv(csv_path)                             │
│                                                              │
│   f) Return result:                                         │
│      { csv_path: "...", gene_count: 1, ... }               │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 6. FRONTEND POLLS STATUS                                    │
├─────────────────────────────────────────────────────────────┤
│ depmap/page.tsx:                                            │
│   - Polls: GET /associations/{job_id}/status               │
│   - Every 3 seconds until status = "finished"              │
│   - When finished: Downloads CSV and displays results       │
└─────────────────────────────────────────────────────────────┘
```

---

## Improvements Over Original Script

| Original Script | DepMap Backend |
|----------------|----------------|
| Command-line only | REST API |
| Runs synchronously | Async background jobs |
| Loads datasets every time | Caches datasets per worker |
| Single output location | Organized by user/experiment |
| No authentication | JWT authentication |
| No user isolation | Per-user file organization |
| Manual execution | Automated via web UI |

---

## Dataset Caching

The worker caches datasets in memory:

```python
_cache = {
    "model": None,           # Loaded once, reused
    "gdsc": None,            # Loaded once, reused
    "ctrp": None,            # Loaded once, reused
    "prism_secondary": None, # Loaded once, reused
    "prism_public": None,    # Loaded once, reused
}
```

**Why this helps:**
- Datasets are large (GB)
- Loading takes time
- Multiple users can share the same worker process
- First job loads datasets; subsequent jobs reuse them

---

## File Organization

```
apps/depmap_backend/outputs/
├── 11/                              # User ID
│   ├── 9de54458-.../                # Experiment ID
│   │   ├── job-abc_associations.csv # Job 1 for gene X
│   │   └── job-def_associations.csv # Job 2 for gene Y
│   └── other-exp-id/                # Different experiment
│       └── job-xyz_associations.csv
└── 12/                              # Different user
    └── same-exp-id/                 # Same experiment ID, different user
        └── job-123_associations.csv
```

This ensures:
- Users can't access each other's files
- Same gene in different experiments = separate files
- Easy to find and manage results

---

## Key Components

### 1. Server (`server.py`)
- FastAPI application entry point
- Configures CORS for frontend
- Registers routers

### 2. API Router (`routers/associations.py`)
- `POST /associations` - Create new analysis job
- `GET /associations/{job_id}/status` - Check job status
- `GET /associations/{job_id}/download` - Download CSV results

### 3. Worker (`workers/depmap_worker.py`)
- Imports functions from `depmap_associations.py`
- Loads and caches datasets
- Processes gene-drug associations
- Saves results to organized directories

### 4. Queue Worker (`workers/queue_worker.py`)
- Sets up Redis queue named "depmap"
- Separate from model_backend's "train" queue
- Uses same Redis instance but isolated jobs

### 5. Authentication (`auth/deps.py`)
- Reuses model_backend's JWT validation
- Ensures users can only access their own jobs

---

## Summary

The transformation:
1. **Extracted** reusable functions from the script
2. **Wrapped** them in a FastAPI microservice
3. **Added** async job processing with Redis
4. **Added** authentication and user isolation
5. **Implemented** dataset caching for performance
6. **Organized** outputs by user and experiment
7. **Created** web UI for easy access

The original script logic remains unchanged; it's now accessible via a web API with better performance, security, and organization.

---

## Running the System

### Start DepMap Backend Server:
```bash
python -m uvicorn apps.depmap_backend.server:app --reload --port 8001
```

### Start DepMap Worker:
```bash
rq worker -u "$REDIS_URL" depmap
```

### Frontend Integration:
- Button on each gene card in experiment details
- Navigates to `/dashboard/depmap?gene=GENE&experimentId=EXP_ID`
- Automatically starts analysis and polls for results

