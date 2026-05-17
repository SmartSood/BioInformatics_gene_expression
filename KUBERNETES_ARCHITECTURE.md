# Gene Web Platform - Kubernetes Architecture

## Table of Contents
1. [System Overview](#system-overview)
2. [Service Architecture](#service-architecture)
3. [Data Flow](#data-flow)
4. [Storage Architecture](#storage-architecture)
5. [Networking](#networking)
6. [Auto-Scaling](#auto-scaling)
7. [Configuration Management](#configuration-management)
8. [Design Decisions](#design-decisions)
9. [Deployment Flow](#deployment-flow)

## System Overview

The Gene Web Platform is deployed on **AWS EKS (Elastic Kubernetes Service)** with a microservices architecture. All services run in the `gene-web` namespace with cloud-native patterns for high availability and automatic scaling.

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     AWS EKS Cluster                          │
│  ┌──────────────────────────────────────────────────────┐   │
│  │            gene-web Namespace                        │   │
│  │                                                      │   │
│  │  ┌─ Frontend Layer ────────────────────────────────┐ │   │
│  │  │ Web Frontend (Next.js)  ← Routes to Auth      │ │   │
│  │  │ Auth Backend (Node.js)  ← JWT/Identity       │ │   │
│  │  └──────────────────────────────────────────────┘ │   │
│  │                    ↓ NGINX Ingress                  │   │
│  │  ┌─ API Layer (ML Backends) ─────────────────────┐ │   │
│  │  │                                               │ │   │
│  │  │  ┌─ Model Backend ─────────────────────────┐ │ │   │
│  │  │  │ API (2 replicas) →  /models, /jobs     │ │ │   │
│  │  │  │ Workers (2 replicas) → Training jobs   │ │ │   │
│  │  │  └─────────────────────────────────────────┘ │ │   │
│  │  │                                               │ │   │
│  │  │  ┌─ Embedding Backend ──────────────────────┐ │ │   │
│  │  │  │ API (2 replicas) →  /embeddings        │ │ │   │
│  │  │  │ Workers (2 replicas) → Async jobs      │ │ │   │
│  │  │  └─────────────────────────────────────────┘ │ │   │
│  │  │                                               │ │   │
│  │  │  ┌─ DepMap Backend ──────────────────────────┐ │ │   │
│  │  │  │ API (2 replicas) →  /associations       │ │ │   │
│  │  │  │ Workers (2 replicas) → Compute jobs     │ │ │   │
│  │  │  └─────────────────────────────────────────┘ │ │   │
│  │  │                                               │ │   │
│  │  │  ┌─ Affinity Backend ──────────────────────┐ │ │   │
│  │  │  │ API (2 replicas) →  /affinity          │ │ │   │
│  │  │  └─────────────────────────────────────────┘ │ │   │
│  │  │                                               │ │   │
│  │  └───────────────────────────────────────────────┘ │   │
│  │                                                      │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
          ↓                    ↓                    ↓
    ┌──────────┐        ┌──────────┐        ┌──────────┐
    │ AWS S3   │        │ ElastiCache      │ AWS RDS  │
    │ Artifacts│        │ Redis (Cache)    │ Postgres │
    └──────────┘        └──────────┘       └──────────┘
```

## Service Architecture

All services are deployed as **Kubernetes Deployments** with load balancing through **Services** and external routing via **NGINX Ingress**.

### Frontend Tier

#### Web Frontend (Next.js)
- **Port**: 3000
- **Replicas**: 2 (min), 5 (max)
- **Purpose**: React-based UI for the platform
- **Endpoints**: `/` (all routes)
- **Auto-scaling**: CPU 70% / Memory 80%

#### Auth Backend (Node.js)
- **Port**: 3001
- **Replicas**: 2 (min), 4 (max)
- **Purpose**: JWT token validation, user identity
- **Endpoints**: `/auth/*`, `/health`
- **Database**: PostgreSQL (RDS)
- **Auto-scaling**: CPU 70% / Memory 80%

### Backend Services Tier

All ML backends follow the **same pattern**:

#### 1. Model Backend (Port 8000)

**API Deployment (2-5 replicas)**
```
POST   /jobs                        Create training job
GET    /jobs/{id}                   Check job status
GET    /experiments                 List training results
GET    /experiments/{id}/download   Download ranked genes CSV
POST   /models/{id}/predict         Run inference
```

**Worker Deployment (2 replicas)**
- Consumes training jobs from Redis queue (RQ)
- Trains ML models using scikit-learn/xgboost
- Saves `model.joblib` and metrics to S3
- Updates database with S3 artifact URIs
- Resources: 500m CPU, 2Gi memory (requests) → 2000m CPU, 8Gi (limits)

**Data Flow**:
```
Training Request → API → Queue in Redis
                              ↓
                         Worker picks up
                              ↓
                         Train model
                              ↓
                         Upload to S3 (s3://gene-web-data/model_backend/user-123/job-456/model.joblib)
                              ↓
                         Update DB with URI
                              ↓
                         Client polls GET /jobs/{id} → Receives S3 URI
```

#### 2. Embedding Backend (Port 8002)

**API Deployment (2-5 replicas)**
```
POST   /embeddings                           Generate embeddings
GET    /embeddings/{id}/artifacts            List artifact types
GET    /embeddings/{id}/download/{format}   Download (CSV/ZIP)
```

**Worker Deployment (2 replicas)**
- Generates drug and gene embeddings using transformers
- Uses pre-trained models (BERT, UniMol)
- Outputs CSVs and ZIP archives
- Uploads all artifacts to S3
- Resources: 1000m CPU, 4Gi memory (requests) → 4000m CPU, 16Gi (limits)

**Artifacts Stored in S3**:
- `input_metadata_csv`
- `drug_embeddings_csv`
- `gene_embeddings_csv`
- `combined_embeddings_csv` (optional)
- `zip_file` (combined archive)

#### 3. DepMap Backend (Port 8001)

**API Deployment (2-4 replicas)**
```
POST   /associations                        Query gene associations
GET    /associations/{gene}/download        Download association CSV
GET    /associations/{job_id}/download      Download job results
```

**Worker Deployment (2 replicas)**
- Loads DepMap dataset (cached in pod memory)
- Computes associations between genes
- Outputs CSV results
- Uploads CSVs to S3 with caching
- Resources: 500m CPU, 2Gi memory (requests) → 2000m CPU, 8Gi (limits)

**Caching Strategy**:
- In-pod dataset cache (loaded once per worker)
- S3 result caching (checks for existing results before recomputing)
- Redis queue for job management

#### 4. Affinity Backend (Port 8003)

**API Deployment (2-4 replicas)**
```
POST   /affinity/predict               Drug-gene affinity prediction
GET    /affinity/results/{id}          Download prediction results
```

**Purpose**: Predicts drug-gene binding affinity using pre-trained models

**Data Flow**:
- Loads checkpoint: `gene_embeddings.pth`
- Generates predictions
- Saves CSV to S3
- Returns S3 URI to client

## Data Flow

### Request Lifecycle

```
1. CLIENT REQUEST
   ↓
2. NGINX INGRESS (with TLS/HTTPS)
   - Routes based on hostname/path
   - Enforces rate limiting (100 req/s)
   - Supports 500MB uploads
   ↓
3. SERVICE (ClusterIP)
   - Internal load balancing
   - DNS: service-name.gene-web.svc.cluster.local
   ↓
4. API POD (1 of N replicas)
   - Validates request
   - Checks authentication (JWT)
   ↓
5. SHORT TASK (< 10s)
   ├─ Process
   ├─ Return result directly
   └─ (e.g., list jobs, get status)
   ↓
6. LONG TASK (> 10s)
   ├─ Validate input
   ├─ Upload input file to S3 (if needed)
   ├─ Create job record in DB
   ├─ Queue job to Redis (RQ)
   ├─ Return job ID to client
   └─ (e.g., train model, compute embeddings)
   ↓
7. WORKER POD
   - Consumes from Redis queue
   - Downloads input from S3 (if needed)
   - Processes job
   - Uploads results to S3
   - Updates DB with S3 URIs
   ↓
8. CLIENT POLLING
   - Polls: GET /jobs/{id}
   - Receives: { status, results: { modelPath: "s3://..." } }
   - Downloads: GET /experiments/{id}/download → downloads from S3
   ↓
9. S3 DIRECT DOWNLOAD
   - FastAPI returns FileResponse from S3
   - Streams file directly to client
```

### Environment Variables Flow

```
GitHub Secrets
    ↓
GitHub Actions (workflow)
    ↓
Update k8s/secrets.yaml with actual values
    ↓
kubectl apply -f k8s/secrets.yaml
    ↓
Kubernetes Secret resource created
    ↓
Pod startup mounts secret as env var
    ↓
Application reads env var (e.g., S3_BUCKET, JWT_SECRET)
```

## Storage Architecture

### Zero Local Persistence Pattern

The platform uses **cloud-native storage** with no persistent volumes:

#### Temporary Storage
- **Location**: Pod temp directory (`/tmp`)
- **Lifecycle**: Deleted when pod terminates
- **Use**: Intermediate files, downloads
- **Example**: 
  ```python
  # Download from S3 to temp
  tmp_path = await download_to_temp(s3_uri)
  # Process file
  process_file(tmp_path)
  # Return to client (temp file cleaned up after response)
  ```

#### Permanent Storage: AWS S3
- **S3 Bucket**: `gene-web-data`
- **All Artifacts**: Models, embeddings, CSVs, predictions
- **Database Stores**: S3 URIs (not local paths)

**Example Artifact Paths**:
```
s3://gene-web-data/
├── model_backend/
│   └── user-123/
│       └── job-456/
│           ├── model.joblib
│           ├── metrics.json
│           └── ranked_genes.csv
├── embedding_backend/
│   └── user-123/
│       └── request-789/
│           ├── input_metadata.csv
│           ├── drug_embeddings.csv
│           ├── gene_embeddings.csv
│           └── combined.zip
├── depmap_backend/
│   └── user-123/
│       └── experiment-101/
│           └── associations.csv
└── affinity_backend/
    └── user-123/
        └── request-202/
            └── predictions.csv
```

#### Cache: AWS ElastiCache (Redis)
- **Purpose**: Job queue (RQ), session cache
- **Endpoints**: `redis-elasticache:6379`
- **Data**: Ephemeral (can be lost without data loss)
- **Queue Format**:
  ```
  RQ Queue: rq:queue:train
  Jobs: { job_id, params, status, result_path }
  ```

#### Database: AWS RDS (PostgreSQL)
- **Connection**: Prisma ORM
- **Tables**:
  - `users` - User accounts
  - `trainingrun` - Training job metadata + S3 URIs
  - `experiment` - Experiment results + S3 URIs
  - `embedding_job` - Embedding tasks
  - `depmap_association` - Gene associations (cached)

**Key Fields Using S3 URIs**:
```sql
-- trainingrun table
modelPath      VARCHAR  -- s3://gene-web-data/model_backend/.../model.joblib
resultsPath    VARCHAR  -- s3://gene-web-data/model_backend/.../ranked_genes.csv

-- embedding_job table
artifact_uris  JSONB    -- { "input": "s3://...", "output_zip": "s3://..." }
```

## Networking

### Service Discovery (Internal)

All services communicate within the cluster using **ClusterIP Services**:

```yaml
Services (DNS Names):
  - model-backend:8000              (API + workers)
  - embedding-backend:8002          (API + workers)
  - depmap-backend:8001             (API + workers)
  - affinity-backend:8003           (API only)
  - auth-backend:3001               (Auth + JWT)
  - web-frontend:3000               (Frontend)
  
Kubernetes DNS:
  - model-backend.gene-web.svc.cluster.local
  - (within cluster, just use: model-backend)
```

**Worker-to-API Communication**:
```python
# Within worker pod, call API for validation
import requests
response = requests.get("http://auth-backend:3001/verify-token")
```

### Ingress Routing (External)

NGINX Ingress Controller routes external traffic based on **hostname + path**:

```yaml
Routing Rules:
  gene-web.example.com                    → web-frontend:3000
  auth.gene-web.example.com               → auth-backend:3001
  
  api.gene-web.example.com/models         → model-backend:8000
  api.gene-web.example.com/jobs           → model-backend:8000
  api.gene-web.example.com/experiments    → model-backend:8000
  
  api.gene-web.example.com/embeddings     → embedding-backend:8002
  
  api.gene-web.example.com/associations   → depmap-backend:8001
  
  api.gene-web.example.com/affinity       → affinity-backend:8003
```

### TLS/HTTPS

- **Certificate Manager**: cert-manager + Let's Encrypt
- **Automatic Renewal**: 30 days before expiry
- **Domains**: `gene-web.example.com`, `auth.gene-web.example.com`, `api.gene-web.example.com`

### Network Policies (Optional)

Can restrict traffic between namespaces/pods if needed:

```yaml
# Only allow ingress from NGINX
# Only allow depmap workers to call Redis
# Deny all traffic by default
```

## Auto-Scaling

### Horizontal Pod Autoscaler (HPA)

Each backend service scales independently based on metrics:

```yaml
Model Backend API HPA:
  - Min Replicas: 2
  - Max Replicas: 5
  - Scale On:
    - CPU > 70% utilization → add replica
    - CPU < 30% utilization → remove replica
    - Memory > 80% utilization → add replica
  - Check Interval: 15 seconds

Embedding Backend API HPA:
  - Min Replicas: 2
  - Max Replicas: 5
  - Same metrics

DepMap Backend API HPA:
  - Min Replicas: 2
  - Max Replicas: 4
  - More conservative (resource-intensive)

Affinity Backend API HPA:
  - Min Replicas: 2
  - Max Replicas: 4
```

**Example Scaling Event**:
```
T=0:     2 replicas running (baseline)
T=10s:   High traffic → CPU spikes to 85%
T=25s:   HPA detects high CPU
T=30s:   New pod starts up
T=40s:   3 replicas running
T=45s:   Traffic normalizes → CPU drops to 60%
T=60s:   No action needed (still above 30%)
T=120s:  Sustained low traffic → CPU 25%
T=135s:  HPA removes replica
T=145s:  2 replicas running (back to baseline)
```

### Worker Scaling

Workers **do not auto-scale** (manual):
```bash
# Scale workers if job queue gets too long
kubectl scale deployment model-backend-worker -n gene-web --replicas=3
```

## Configuration Management

### ConfigMap (Non-sensitive config)

```yaml
shared-config:
  REDIS_URL: redis://redis-elasticache:6379/0
  USE_S3: "true"
  S3_REGION: us-east-1
  PYTHONUNBUFFERED: "1"

model-backend-config:
  MODEL_BACKEND_PORT: "8000"

embedding-backend-config:
  EMBEDDING_BACKEND_PORT: "8002"
  HF_HOME: /tmp/hf_cache
```

**Usage in Pod**:
```yaml
containers:
  - name: api
    envFrom:
      - configMapRef:
          name: shared-config
      - configMapRef:
          name: model-backend-config
```

### Secrets (Sensitive config)

```yaml
aws-credentials:
  S3_BUCKET: gene-web-data
  S3_ACCESS_KEY: <encrypted>
  S3_SECRET_KEY: <encrypted>

jwt-secret:
  JWT_SECRET: <encrypted>
  JWT_ALGORITHM: HS256

database-credentials:
  DATABASE_URL: postgresql://user:pass@rds.amazonaws.com:5432/gene_web

redis-credentials:
  REDIS_URL: redis://redis-elasticache:6379/0
```

**Updating Secrets**:
```bash
# Edit and reapply
kubectl edit secret aws-credentials -n gene-web

# Or replace entirely
kubectl delete secret aws-credentials -n gene-web
kubectl create secret generic aws-credentials \
  --from-literal=S3_BUCKET=new-value \
  -n gene-web

# Force pod restart to pick up new secrets
kubectl rollout restart deployment/model-backend-api -n gene-web
```

## Design Decisions

### 1. Separation of API and Workers

**Why?**
- Independent scaling: API handles HTTP traffic, workers handle long jobs
- Resource isolation: Workers can be tuned for CPU-heavy tasks
- Fault isolation: Worker crash doesn't affect API availability

**Example**:
```
High traffic day:
  - API scales to 5 replicas (serving requests fast)
  - Workers stay at 2 replicas (jobs still processing normally)

No traffic, but long queue:
  - Scale workers to 5 replicas (speed up queue processing)
  - API stays at 2 replicas (no requests)
```

### 2. No Persistent Volumes

**Why?**
- Pods are ephemeral, stateless, and disposable
- Easy to scale horizontally
- Easy to update (rolling updates work without data concerns)
- Lower cost (no EBS volumes sitting idle)
- S3 is more durable than local storage

**Trade-off**: Download/upload latency to S3 (< 100ms typically)

### 3. S3-First Storage Model

**Why?**
- All replicas see same data (no sync issues)
- Survives pod deletion
- Searchable via AWS CLI
- Can be accessed from outside the cluster
- Integrates with data lake/analytics tools

**URIs in Database**:
```python
# Instead of: /artifacts/job-123/model.joblib
# Store: s3://gene-web-data/model_backend/user-123/job-123/model.joblib

# Advantages:
# - Direct S3 link, no pod needed to download
# - Works across deployments
# - Easy to migrate/backup
```

### 4. Multiple Replicas by Default

**Why?**
- High availability (if one pod dies, others keep serving)
- Zero-downtime deployments (old pod continues while new pod starts)
- Load distribution
- Resource utilization (multiple replicas better than one big pod)

**Pod Disruption Budget** (future):
```yaml
minAvailable: 1  # Always keep at least 1 pod running
```

### 5. Namespace Isolation

**Why?**
- All resources in `gene-web` namespace
- Easy to manage (delete whole namespace = clean up everything)
- Can have other namespaces for other projects
- RBAC policies scoped per namespace

**Other Namespaces**:
```
kube-system          (Kubernetes internals)
ingress-nginx        (NGINX controller)
cert-manager         (Let's Encrypt)
monitoring           (Prometheus/Grafana - optional)
```

## Deployment Flow

### From Git Push to Running Pods

```
1. DEVELOPER PUSH
   git push origin main
   ↓
2. GITHUB WEBHOOK
   Triggers GitHub Actions workflow
   ↓
3. BUILD STAGE (parallel jobs)
   ├─ Build model-backend image
   ├─ Build embedding-backend image
   ├─ Build depmap-backend image
   ├─ Build affinity-backend image
   ├─ Build auth-backend image
   └─ Build web-frontend image
   ↓
4. PUSH TO DOCKER HUB
   docker.io/username/gene-model:SHA123
   docker.io/username/gene-embedding:SHA123
   ... (all tagged with commit SHA and "latest")
   ↓
5. UPDATE MANIFESTS
   sed -i 's/:latest/:SHA123/g' k8s/*.yaml
   ↓
6. DEPLOY TO EKS
   kubectl apply -f k8s/namespace.yaml
   kubectl apply -f k8s/configmap.yaml
   kubectl apply -f k8s/secrets.yaml
   kubectl apply -f k8s/model-backend.yaml
   kubectl apply -f k8s/embedding-backend.yaml
   kubectl apply -f k8s/depmap-backend.yaml
   kubectl apply -f k8s/affinity-backend.yaml
   kubectl apply -f k8s/frontend-auth.yaml
   kubectl apply -f k8s/ingress.yaml
   ↓
7. ROLLING UPDATE (per deployment)
   Old pods:                     New pods:
   ├─ model-backend-api-abc     └─ model-backend-api-xyz (starting)
   ├─ model-backend-api-def        model-backend-api-xyz (ready)
   └─ model-backend-api-ghi     ↓
   
   Old pods:                     New pods:
   ├─ model-backend-api-def     ├─ model-backend-api-abc (terminating)
   └─ model-backend-api-ghi        model-backend-api-xyz
   
   Old pods: (terminated)        New pods:
                                 ├─ model-backend-api-def
                                 └─ model-backend-api-xyz
   ↓
8. HEALTH CHECKS
   kubectl wait --for=condition=available deployment/... -n gene-web
   ↓
9. MONITORING
   - Pod metrics (CPU, memory)
   - Service endpoints (ready/not-ready)
   - Application logs (stdout/stderr)
   ↓
10. TRAFFIC FLOWS TO NEW PODS
    ↓ (old version replaced, zero downtime)
```

### Rollback Process

```
If deployment fails:

1. kubectl rollout undo deployment/model-backend-api -n gene-web
   ↓ (reverts to previous image tag)

2. kubectl rollout status deployment/model-backend-api -n gene-web
   ↓ (waits for rollback to complete)

3. Traffic flows to previous version
   ↓ (old pods restarted with previous image)

Rollback history:
kubectl rollout history deployment/model-backend-api -n gene-web
revision 1: image: gene-model:abc123
revision 2: image: gene-model:def456  ← current
revision 3: image: gene-model:ghi789  ← will rollback to here

kubectl rollout undo deployment/model-backend-api --to-revision=1 -n gene-web
```

## Resource Usage Summary

### CPU/Memory Requests and Limits

```
Frontend:
  - Web Frontend:      250m / 512Mi  →  1000m / 2Gi
  - Auth Backend:      250m / 512Mi  →  1000m / 2Gi

Backends:
  - Model API:         500m / 1Gi    →  2000m / 4Gi
  - Model Worker:      500m / 2Gi    →  2000m / 8Gi
  - Embedding API:     1000m / 4Gi   →  4000m / 16Gi
  - Embedding Worker:  1000m / 4Gi   →  4000m / 16Gi
  - DepMap API:        500m / 2Gi    →  2000m / 8Gi
  - DepMap Worker:     500m / 2Gi    →  2000m / 8Gi
  - Affinity API:      500m / 2Gi    →  2000m / 8Gi

Total Baseline (requests):
  ≈ 5000m CPU (5 cores)
  ≈ 27Gi Memory
```

## Monitoring & Observability

### Key Metrics to Monitor

```
Pod Metrics:
  - CPU utilization (↑ = scale up)
  - Memory utilization (↑ = leak? or normal)
  - Pod restart count (↑ = crash loop)
  - Pod ready time (↑ = slow startup)

Service Metrics:
  - Request rate (requests/sec)
  - Response time (latency)
  - Error rate (4xx, 5xx responses)
  - Queue depth (RQ jobs waiting)

Cluster Metrics:
  - Node CPU/memory available
  - Pod placement efficiency
  - Network I/O
  - S3 API call rate
  - RDS connection count
```

### Observability Stack (Optional)

```yaml
namespace: monitoring

Prometheus:
  - Scrapes pod metrics
  - Stores in time-series DB
  - Alerting rules

Grafana:
  - Dashboard visualization
  - Custom alerts

ELK Stack (or CloudWatch):
  - Pod logs aggregation
  - Structured logging (JSON)
  - Search and analysis
```

## Troubleshooting Guide

### Common Issues

**1. Pod stuck in Pending**
```bash
kubectl describe pod <pod-name> -n gene-web
# Check: insufficient CPU/memory, node selector, PVC binding
```

**2. Pod CrashLoopBackOff**
```bash
kubectl logs <pod-name> -n gene-web
# Check: application error, missing env var, import error
```

**3. Service not reachable**
```bash
kubectl get endpoints <service-name> -n gene-web
# Check: endpoints empty (pods not ready)
kubectl logs -l app=model-backend -n gene-web
```

**4. High latency**
```bash
# Check S3 latency
time aws s3 ls s3://gene-web-data/

# Check RDS latency
psql -h rds-endpoint -U postgres -d gene_web -c "SELECT 1"

# Check Redis latency
redis-cli -h redis-elasticache ping
```

**5. OOM (Out of Memory)**
```bash
# Check which pod/container
kubectl top pod -n gene-web

# Increase memory limits in manifest
# Restart deployment
```

---

**Generated**: May 16, 2026
**Architecture Version**: 1.0
**Platform**: AWS EKS + S3 + RDS + ElastiCache
**Languages**: Python (FastAPI), Node.js (Express/Next.js), Kubernetes YAML
