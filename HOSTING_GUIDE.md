# Gene Web — Hosting Guide

Reference doc for the live AWS deployment: what's running, why it's built this way, how to operate it, and the full history of bugs found and fixed while getting it working. Written so this is understandable on its own, without needing the chat history that produced it.

## 1. The architecture, in one picture

```
                                   ┌──────────────────────────────┐
                                   │  Tier 2 — on-demand (k3s)    │
                                   │  t4g.xlarge, Elastic IP       │
      user's browser               │  13.126.245.2                 │
           │                       │  STOPPED by default            │
           ▼                       │                                 │
┌──────────────────────┐  StartInstances   model/embedding/depmap/  │
│ Tier 1 — always on    │──────────────▶│  affinity backends + Redis │
│ 13.207.189.132         │               │  (own k3s namespace)      │
│ t4g.small, PM2+Nginx   │  idle timer   │                            │
│                        │◀──stops itself│  Traefik ingress           │
│ web-frontend            │              └─────────────┬──────────────┘
│ auth-backend             │                             │
│ wake-gateway  ───────────┼─────────────────────────────┘
└──────────────────────┘        real API calls, once ready
```

**Tier 1** is what's actually up 24/7: the landing page, login, and a small `wake-gateway` service. Cheap (~$12/mo on-demand), because it's genuinely lightweight — no ML, no heavy compute.

**Tier 2** is a real k3s cluster (genuine Kubernetes — same `kubectl`, same `Deployment`/`Service`/`HPA` specs as EKS) that sits **stopped** most of the time. The frontend calls `wake-gateway` before any ML feature; it starts Tier 2 via the EC2 API, the frontend polls until it reports ready (health-checked through the real ingress, not just "instance running"), then the real API call goes through. A systemd timer on Tier 2 checks Traefik's own access logs every 5 minutes and stops the instance itself after ~20 minutes with no real traffic.

Because Tier 2 only bills for the hours it's actually running, its instance size barely affects the monthly bill — it's sized for headroom (`t4g.xlarge`, 16GB RAM) rather than cost, since a `t4g.large` vs `t4g.xlarge` difference is a few cents/month at this usage level.

## 2. What's deployed where

| | Value |
|---|---|
| Tier 1 instance ID | `i-01e3ad75affbfe66c` |
| Tier 1 public IP | `13.207.189.132` (dynamic — no Elastic IP; fine since it never stops) |
| Tier 2 instance ID | `i-0cb21b1b889faa628` |
| Tier 2 public IP | `13.126.245.2` (**Elastic IP** `eipalloc-0fbe238beaad1f768` — this is fixed permanently, survives stop/start) |
| Region | `ap-south-1` (Mumbai — chosen over Hyderabad `ap-south-2` for full Graviton/`t4g` availability) |
| Security group | `sg-03bcce6e7fb5de035` (`gene-web-sg`) — 80/443 open to the world, 22 restricted to your IP + AWS's EC2 Instance Connect range (`13.233.177.0/29`) |
| SSH key pair | `gene-web-ec2-key` — private key at `~/.ssh/gene-web-ec2-key.pem` on your Mac |
| S3 bucket | `gene-web-data` (ap-south-1) — dataset uploads, training outputs, embedding results |
| Docker registry | Docker Hub, `smarthcnbl/gene-*` (public repos: `gene-model`, `gene-embedding`, `gene-depmap`, `gene-affinity`, `gene-auth`, `gene-web`, `shared-base`) |
| Container orchestration | k3s (CNCF-certified upstream Kubernetes, single control-plane node) on Tier 2 only |

**Tier 2's public IP will never change again** — it's an Elastic IP, not the ephemeral IP EC2 assigns by default. (Early in setup it *did* change once on a stop/start, which broke everything hardcoded to the old IP — that's exactly why the Elastic IP was added. See §5.)

## 3. How to get in

Everything below assumes your local network may be blocking outbound SSH (port 22) — this happened repeatedly during setup. If a plain `ssh` command hangs, use **AWS Console → EC2 → select instance → Connect → EC2 Instance Connect** instead (browser-based, works around local network SSH blocks since it doesn't use your machine's port 22 at all).

```bash
ssh -i ~/.ssh/gene-web-ec2-key.pem ubuntu@13.207.189.132   # Tier 1
ssh -i ~/.ssh/gene-web-ec2-key.pem ubuntu@13.126.245.2     # Tier 2
```

On Tier 2, `kubectl` is `k3s kubectl`, and everything needs `sudo` (kubeconfig is root-only):

```bash
sudo k3s kubectl get pods -n gene-web
sudo k3s kubectl logs -n gene-web deployment/model-backend-api --tail=50
```

Repo is checked out at `/opt/gene-web` on both boxes (root-owned — use `sudo git pull`, `sudo` for file edits).

## 4. The deploy loop (no managed CI/CD)

Everything is deployed manually via a consistent loop — build locally, push to Docker Hub, pull + apply on the server:

```bash
# 1. Make a code/config change locally, commit + push to GitHub main
git add <files> && git commit -m "..." && git push main main   # remote is named "main", not "origin"

# 2. If it's a CODE change (not just k8s YAML), rebuild + push images
DOCKER_USERNAME=smarthcnbl PUSH_IMAGES=true LOAD_TO_K3S=false bash scripts/build-and-push-images.sh
# tags every image with the current git short SHA, e.g. smarthcnbl/gene-model:706a4e0

# 3. Update k8s/tier2/*.yaml to reference the new tag (sed one-liner, see git log for the pattern), commit + push

# 4. On Tier 2: pull the repo, apply the changed manifests
cd /opt/gene-web && sudo git pull origin main
sudo k3s kubectl apply -f k8s/tier2/model-backend.yaml   # etc, whichever changed
sudo k3s kubectl rollout status deployment/model-backend-api -n gene-web --timeout=90s

# 5. On Tier 1, for frontend/auth changes: rebuild in ONE root shell (sudo alone drops sourced env vars)
sudo bash -c 'set -a; source .env; set +a; npm run build --workspace apps/web'
sudo pm2 restart gene-web-frontend --update-env
```

**Config-only changes** (ConfigMap/Secret/resource limits) don't need an image rebuild — just edit the YAML, commit, pull on Tier 2, `kubectl apply`, and `kubectl rollout restart` if it's a Secret/ConfigMap change (editing a Secret doesn't auto-propagate to already-running pods).

### Idle-checker gotcha

**Before doing any extended debugging session on Tier 2, pause the auto-stop:**
```bash
sudo systemctl stop idle-checker.timer
```
Otherwise it can stop the instance mid-session (kubectl/SSH activity doesn't count as "traffic" — only real requests through the public Traefik ingress do). Re-enable when done:
```bash
sudo systemctl start idle-checker.timer
```

## 5. Secrets and config — what lives where

- **`k8s/tier2/secrets.yaml`** in git only ever contains the `jwt-secret` *placeholder*. **`database-credentials` is deliberately never in a git-tracked file** — it must be applied directly every time:
  ```bash
  kubectl create secret generic database-credentials -n gene-web \
    --from-literal=DATABASE_URL='<real Neon URL>' \
    --dry-run=client -o yaml | kubectl apply -f -
  ```
  (A prior version had it in the file, and a routine `kubectl apply -f secrets.yaml` for an unrelated fix silently reset it to a placeholder — a real production outage. Don't reintroduce it there.)
- **`JWT_SECRET`** must be byte-identical between Tier 1 (`/opt/gene-web/.env`, used by `auth_backend` to sign tokens) and Tier 2 (k8s Secret, used by all 4 backends to verify them). To compare without ever printing the actual value:
  ```bash
  # Tier 1:
  sudo grep '^JWT_SECRET=' /opt/gene-web/.env | cut -d= -f2- | tr -d "\"'\r\n" | sha256sum
  # Tier 2:
  sudo k3s kubectl get secret jwt-secret -n gene-web -o jsonpath='{.data.JWT_SECRET}' | base64 -d | sha256sum
  ```
  Matching hashes = matching secret.
- **`AUTH_JWT_ISSUER`/`AUTH_JWT_AUDIENCE`** (in Tier 2's `shared-config` ConfigMap) must exactly match the literal strings `auth_backend` hardcodes when signing (`apps/auth_backend/src/index.ts`: issuer `"http://localhost:4000"`, audience `"mlapp"` — yes, the issuer string looks like a dev artifact, but it's what's actually signed, so verification has to match it exactly).
- **Tier 1's `.env`** holds: `FE_PORT`, `AUTH_PORT`, `WAKE_GATEWAY_PORT`, `WAKE_SHARED_SECRET`, `AWS_REGION`, `TIER2_INSTANCE_ID`, `TIER2_HEALTH_URL`, all the `NEXT_PUBLIC_*_URL` vars (baked into the frontend at *build* time, not read live — changing them requires a rebuild), plus the usual `DATABASE_URL`/`JWT_SECRET`/`SALT_ROUNDS`.
- **`CORS_ALLOWED_ORIGINS`** (Tier 2 `shared-config`) must include Tier 1's origin (`http://13.207.189.132`) — the frontend and Tier 2's APIs are genuinely different origins in production.

## 6. IAM — what has access to what

Three IAM identities matter here, all intentionally least-privilege:

1. **Your own IAM user (`smarth`)** — used for `aws` CLI deploy operations (launching/stopping instances, creating security groups, managing the two roles below, S3 bucket, Elastic IP). Policy at `infra/iam/deploy-user-policy-merged.json` (this is what's actually attached — the account has a 2048-char inline-policy limit, so it's a merged/minified version of what was originally several separate policy files under `infra/iam/`).
2. **`gene-web-tier1-wake`** (Tier 1's instance role) — can only call `ec2:StartInstances` on the specific instance tagged `Role=gene-web-tier2`. Used by `wake-gateway`.
3. **`gene-web-tier2-self-stop`** (Tier 2's instance role) — can only call `ec2:StopInstances` on itself (same tag condition), plus S3 access scoped to just the `gene-web-data` bucket (`infra/iam/tier2-s3-datasets-policy.json`). Used by `idle-checker.sh` and by every backend's `boto3` client (which has no explicit AWS keys configured — it relies entirely on this instance role via the EC2 metadata service).

`ec2:AssociateAddress`/`DisassociateAddress` (for the Elastic IP) and `PassRole` are both scoped with `Condition` blocks so a compromised deploy user still can't do anything outside this exact scope.

## 7. Everything that was broken and fixed, in order

This is the actual debugging history — useful both to understand *why* the code looks the way it does in a few places, and as a checklist if something regresses.

1. **`INSTALL_K3S_VERSION` vs `INSTALL_K3S_CHANNEL`** — k3s install script needs a channel name via the latter env var, not a bare version string via the former.
2. **CUDA bloat in `shared-base.Dockerfile`** — plain `pip install torch` pulls a ~2GB CUDA build on Linux even though Tier 2 has no GPU. Fixed via `--index-url https://download.pytorch.org/whl/cpu`. Same issue with `xgboost` (hard-declares `nvidia-nccl-cu13` on *any* Linux platform, GPU or not) — fixed via `--no-deps` since its real deps (numpy, scipy) are already covered elsewhere.
3. **Missing system libs in the runtime image** — `libgomp1`/`libxrender1`/`libxext6`/`libexpat1` needed by `rdkit.Chem.Draw` just to *import*, and `libatomic1` needed by Prisma's bundled Node CLI at build time. The original Dockerfile ran zero `apt-get` in the runtime stage.
4. **Prisma Python client never generated** — `from prisma import Prisma` failed with "Client hasn't been generated yet". Fixed by running `prisma generate --generator py_client` in the Docker build.
5. **Prisma query engine binary missing at runtime** — `generate` only produces the client's Python *source*; the actual query engine is a separate platform-specific binary fetched via `prisma py fetch` into a cache dir that never got copied from the build stage into the runtime image.
6. **Prisma engine connect timeout too short** — the query engine's cold-start on this hardware genuinely takes slightly over the default ~10s timeout. Fixed via `db.connect(timeout=30)`.
7. **`python-multipart` missing** — needed by FastAPI for any file-upload endpoint, not pulled in automatically.
8. **Cross-service imports never copied** — `embedding_backend`/`depmap_backend`/`affinity_backend` all import `apps.model_backend.auth` and `.storage` (shared JWT/S3 helpers), and `depmap_backend` also imports `model_backend`'s `client/db.py` (as bare `client`, matching its own `PYTHONPATH`) — none of these were ever `COPY`'d into those services' Docker images.
9. **Bundled data/model directories never shipped** — `depmap_backend` needs a sibling `apps/depmap` (1.2GB public reference dataset), `embedding_backend` needs `apps/embedding_bundle` (490MB of Mol2Vec/GIN code + pretrained weights) — both accessed via runtime `sys.path` tricks, neither ever `COPY`'d in.
10. **`.dockerignore` silently stripped the embedding models** — broad `models`/`**/models`/`*.pt`/`*.pkl` excludes (meant for local dev caches) also excluded the *required* `apps/embedding_bundle/models/*.pt`/`.pkl` files from every build context. Fixed with `!apps/embedding_bundle/models/**` negation.
11. **Hardcoded `localhost` URLs everywhere** — `packages/config` (frontend backend URLs), a duplicated set in `NewExperimentForm.tsx`, three more in the embeddings page, and all 4 FastAPI backends' CORS `allow_origins`. All switched to env-driven with `localhost` as the dev fallback.
12. **`@repo/dotenv-path`/`@repo/db`/`@repo/zod-scemma` export raw `.ts`** — fine for Next.js (transpiles workspace packages), fatal for plain `node dist/index.js` (`ERR_UNKNOWN_FILE_EXTENSION`). Fixed by running `auth_backend`/`wake_gateway` via `tsx` in production instead of the compiled output.
13. **JS Prisma client committed with a macOS binary** — `packages/db/generated/prisma` was generated on the dev machine (darwin-arm64), doesn't work on Tier 1's Linux/arm64. Fixed by regenerating natively in `infra/tier1/userdata.sh`.
14. **Tier 2's public IP changing on stop/start** — see §2; fixed with an Elastic IP.
15. **Idle-checker counting its own noise** — Traefik's `--ping=true` self-health-check hit `/ping` every few seconds, permanently looking like "real traffic" and defeating the entire auto-stop design. Excluded from the count.
16. **Ingress missing most of `model-backend`'s real routes** — only `/models`/`/jobs`/`/experiments` were routed; the actual FastAPI routers also use `/health`, `/dataset`, `/datasets`, `/train` (verified against each router's actual `prefix=` — Traefik does no path rewriting, so every real path needs its own rule).
17. **Zero JWT config on Tier 2** — `JWT_SECRET`/`AUTH_JWT_ISSUER`/`AUTH_JWT_AUDIENCE` were never wired in at all; every authenticated request 500'd on a "missing server configuration" check before even trying to verify anything.
18. **S3 bucket never actually created**, and **Tier 2's IAM role had zero S3 permissions** — the config referenced `gene-web-data` as if it existed; it didn't, and even after creating it the role couldn't touch it.
19. **S3 region mismatch** — `S3_REGION` was left at a `us-east-1` placeholder while the bucket is in `ap-south-1`. `boto3` tolerated this (redirect handling); `s3fs` (used for direct `pd.read_csv("s3://...")` reads) did not — hard `403`. Also added the standard `AWS_DEFAULT_REGION`/`AWS_REGION` env vars, which `s3fs` actually reads (the custom `S3_REGION` name is only understood by this repo's own `s3_storage.py`).
20. **RQ workers listening on the wrong queue** — `train.py`/`associations.py`/`embeddings.py` each enqueue to their own named queue (`"train"`/`"depmap"`/`"embedding"`), but every worker deployment ran bare `rq.cli worker` with no queue name, which only listens on `"default"`. Every job ever submitted sat untouched forever. Fixed by pointing each worker's `command:` at the purpose-built entrypoint script that already existed for exactly this (`apps/*/workers/{rq_worker,run_rq_worker}.py`).
21. **`model_backend`'s worker script used a removed RQ API** — `from rq import Connection` was removed in RQ 2.x; would have crashed on startup even after the queue-name fix.
22. **`MLFLOW_ALLOW_FILE_STORE`** — newer MLflow versions hard-block plain filesystem tracking (used here deliberately, single-node, no DB-backed tracking server needed) unless explicitly opted in.
23. **`depmap_associations.py`'s `DATASET_DIR` was a bare relative string** (`"dataset"`), only resolving correctly if the process's cwd happened to match the script's own directory — true only via a hypothetical direct invocation, never true for the actual worker process (runs from `/app`). Every gene lookup silently read from a path that didn't exist and reported it as "gene not found" rather than "file not found". Anchored to the script's own `__file__` location instead.
24. **`depmap-backend-worker` OOMKilled on every real job** — a 1Gi memory limit against a workload that loads a 543MB CSV (several times that in memory once parsed) plus multiple other bundled datasets. Bumped to a 6Gi limit (cheap since this only runs during actual bursty job processing, not sustained).
25. **`depmap-backend-worker` missing `DATABASE_URL`** — only the API container had it wired in from an earlier fix; the worker (which actually writes results back after computing them) didn't, so every job "succeeded" per RQ but silently failed to persist its results.
26. **Broken redundant ownership check on CSV download** — `/associations/{job_id}/download` string-matched the user ID against the local temp file path as an "extra security" check, but `download_to_temp()` only keeps the S3 key's basename (e.g. `results.csv`), stripping the `<user_id>/<job>/...` prefix that carried the ID — so it could never pass for any S3-backed result. Removed; the real ownership check (comparing the job's own recorded args to the requesting user) was already correct and unaffected.

### Known outstanding issue (not yet fixed as of this doc)

**Embedding generation fails**: `apps/embedding_backend/services/embedding_service.py` loads ESM2/ProtBERT via `AutoTokenizer.from_pretrained(..., local_files_only=True)` — hardcoded to *never* download, only use a local HuggingFace cache. That cache (`apps/embedding_bundle/hf_cache/`) is only ~580KB, nowhere near enough to hold real pretrained transformer weights (ESM2-650M alone is ~1.3GB) — it was never actually populated. Needs either: pre-downloading these models into that cache and re-bundling (mirrors the `.dockerignore` issue in §7.10 — check that models directory isn't excluded either), or relaxing `local_files_only=True` to allow a one-time download on first use (trades a slow cold start for not needing to ship multi-GB of transformer weights in the image).

## 8. Cost

| Item | Monthly |
|---|---|
| Tier 1, on-demand 24/7 (`t4g.small`) | ~$12 |
| Tier 2 EBS storage while stopped (~40GB gp3) | ~$3 |
| Tier 2 compute, actual usage (`t4g.xlarge`, bursty) | ~$2-5 |
| Elastic IP (only billed while Tier 2 is stopped — free while running) | ~$3-4 |
| S3 (datasets, small) | ~$1 |
| **Total** | **~$21-24/month** (₹1,750-2,000) |

## 9. Quick reference — common commands

```bash
# Pod status / logs
sudo k3s kubectl get pods -n gene-web
POD=$(sudo k3s kubectl get pods -n gene-web -l app=<name>,component=<api|worker> -o jsonpath='{.items[0].metadata.name}')
sudo k3s kubectl logs -n gene-web "$POD" --tail=50 -f

# Resource usage (check for OOM risk)
sudo k3s kubectl top pod -n gene-web
sudo k3s kubectl describe pod -n gene-web -l app=<name> | grep -A8 "Last State\|OOMKilled"

# Redis job queues
sudo k3s kubectl exec -n gene-web deployment/redis-elasticache -- redis-cli KEYS "rq:*"
sudo k3s kubectl exec -n gene-web deployment/redis-elasticache -- redis-cli LLEN rq:queue:<name>

# Force a full redeploy after any manifest change
sudo k3s kubectl apply -f k8s/tier2/<file>.yaml
sudo k3s kubectl rollout restart deployment -n gene-web   # picks up ConfigMap/Secret changes too

# Manually wake / check Tier 2 from Tier 1
curl -s -X POST http://localhost/wake -H "x-wake-secret: $(sudo grep WAKE_SHARED_SECRET .env | cut -d= -f2-)"
curl -s http://localhost/wake/status

# PM2 (Tier 1)
sudo pm2 list
sudo pm2 logs gene-web-frontend --lines 30
sudo pm2 restart gene-wake-gateway --update-env
```

## 10. Full tech stack

### Frontend
- Next.js 15 (App Router) + React 19, TypeScript
- Tailwind CSS 4
- Axios (HTTP client)
- Turborepo (monorepo build orchestration, npm workspaces)

### Backend — Node.js
- Express.js + TypeScript (`auth_backend`)
- `wake_gateway` (custom Express/TypeScript control-plane service)
- Prisma ORM (JS client) + PostgreSQL
- JWT (`jsonwebtoken`) + `bcrypt` for auth
- Run via `tsx` in production (not compiled output — see §7.12)

### Backend — Python / ML
- FastAPI + Uvicorn (`model_backend`, `embedding_backend`, `depmap_backend`, `affinity_backend`)
- Prisma ORM (Python client)
- Redis + RQ (async job queue, one named queue per service)
- MLflow (experiment tracking, filesystem-backed)
- pandas, NumPy, scikit-learn, XGBoost
- PyTorch (CPU build), Transformers (HuggingFace), RDKit
- gensim (Word2Vec — Mol2Vec, ProtVec), unimol-tools
- boto3 + s3fs (S3 access)

### ML models
- Tabular: Random Forest, SVM, MLP, Gradient Boosting, Logistic Regression, XGBoost
- Protein/gene embeddings: ESM-2, ProtBERT, ProtVec
- Drug/compound embeddings: Mol2Vec, GIN/GROVER, Uni-Mol
- Drug-target affinity: custom PyTorch CNN (checkpoint-based inference)

### Data & storage
- PostgreSQL (Neon, serverless) — application data via Prisma
- Redis — job queues
- AWS S3 — datasets, training outputs, embedding/affinity results
- DepMap public reference datasets (expression, GDSC, CTRP, PRISM) — bundled in-image

### Infrastructure & DevOps
- AWS EC2 (Graviton/ARM64, `t4g` family)
- Docker (multi-stage builds, shared base-image strategy)
- Kubernetes (k3s — CNCF-certified upstream k8s)
- Traefik (Tier 2 ingress), Nginx (Tier 1 reverse proxy)
- PM2 (Tier 1 process manager)
- AWS IAM (least-privilege, resource-tag-scoped policies)
- AWS VPC, Security Groups, Elastic IP
- Docker Hub (container registry)
- systemd (timers/services — idle-checker, PM2 boot persistence)
- Git/GitHub (manual GitOps-style deploy loop, no managed CI/CD)

### External APIs / data sources
- PubChem, UniProt, RCSB PDB (compound/protein lookups)
- DepMap (public cancer dependency map datasets)

