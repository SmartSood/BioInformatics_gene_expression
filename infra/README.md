# Two-Tier AWS Deployment (wake-on-demand)

Cheap, always-on presence + real k8s compute that only runs (and bills) when
someone actually uses it.

```
                                   ┌─────────────────────────────┐
                                   │  Tier 2 - on-demand (k3s)   │
                                   │  t4g.xlarge, STOPPED by      │
      user's browser               │  default                     │
           │                       │                               │
           ▼                       │  model / embedding / depmap / │
┌──────────────────────┐  StartInstances  affinity backends        │
│ Tier 1 - always on    │──────────────▶│  Postgres, Redis          │
│ t4g.small (PM2+Nginx) │               │  (own k3s namespace)      │
│                        │  idle timer   │                          │
│ web-frontend           │◀──stops itself│  Traefik ingress          │
│ auth-backend           │               └─────────────┬────────────┘
│ wake-gateway  ─────────┼───────────────────────────────┘
└──────────────────────┘        real API calls, once ready
```

- **Tier 1** is what's actually "up 24/7": the landing page, login, and a
  small `wake-gateway` API. Cheap enough to just leave running.
- **Tier 2** is a real k3s cluster (genuine Kubernetes, not a toy) that sits
  **stopped** most of the time. The frontend calls `wake-gateway` before any
  ML feature; it starts Tier 2, the frontend polls until it's ready, then
  calls the real API. A systemd timer on Tier 2 stops it again after ~20min
  idle.
- Because Tier 2 only bills for the hours it's actually running, its
  **instance size barely affects your monthly bill** — so it's sized for
  headroom (running 3-4 side projects' namespaces at once), not for cost.

## Why this is a legitimate k8s story, not a shortcut

k3s is upstream, CNCF-certified Kubernetes — same `kubectl`, same
`Deployment`/`Service`/`HPA` specs as EKS. Everything in [`k8s/tier2/`](../k8s/tier2/)
ports to EKS unchanged later; only the StorageClass (`local-path` → EBS CSI),
ingress controller (Traefik → nginx/ALB), and Postgres/Redis (in-cluster →
RDS/ElastiCache) would need to change. See the k8s/ vs k8s/tier2/ split below.

## Instance sizing

| | Instance | Why |
|---|---|---|
| Tier 1 | `t4g.small` (2 vCPU, 2GB) | Runs 24/7 — size drives cost directly. 2GB covers frontend + auth + wake-gateway with room to spare (previous 1GB/`t4g.micro` attempts got cramped once Node + PM2 + Nginx overhead stacked up). Resize to `t4g.medium` (4GB, +~$12/mo) later with `stop → modify instance type → start`, no re-architecture needed. |
| Tier 2 | `t4g.xlarge` (4 vCPU, 16GB) | Off by default, so size barely affects the bill (see cost table). 16GB comfortably fits gene-web's ML backends (~6GB requests) *and* leaves room for 2-3 other projects' namespaces to sit alongside it. |

## Cost estimate (₹1,000–2,000/month target)

| Item | Monthly |
|---|---|
| Tier 1, on-demand 24/7 (`t4g.small`) | ~$12 |
| Tier 2 EBS storage while stopped (~40GB gp3) | ~$3 |
| Tier 2 compute, ~15hrs/month actual usage (`t4g.xlarge`) | ~$2 |
| S3 (datasets, small) | ~$1 |
| **Total** | **~$18 (₹1,500)** |

Comfortably inside budget, inside your $75 credit for the first ~4 months
even before it's "free."

## Setup order

### 1. Launch Tier 2 first (you need its instance ID for Tier 1's config)

- AMI: Ubuntu 22.04 LTS, `t4g.xlarge`, 40GB gp3
- **Tag it `Role=gene-web-tier2`** — both IAM policies below key off this tag
- User data: [`infra/tier2/userdata.sh`](tier2/userdata.sh)
- Security group: 80/443 open (API access once awake), 22 restricted to your IP
- Note the instance ID (`i-xxxxxxxx`) once launched

### 2. Create the two IAM roles

```bash
aws iam create-role --role-name gene-web-tier1-wake \
  --assume-role-policy-document '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"ec2.amazonaws.com"},"Action":"sts:AssumeRole"}]}'
aws iam put-role-policy --role-name gene-web-tier1-wake \
  --policy-name wake-tier2 --policy-document file://infra/iam/tier1-wake-gateway-policy.json
aws iam create-instance-profile --instance-profile-name gene-web-tier1-wake
aws iam add-role-to-instance-profile --instance-profile-name gene-web-tier1-wake --role-name gene-web-tier1-wake

aws iam create-role --role-name gene-web-tier2-self-stop \
  --assume-role-policy-document '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"ec2.amazonaws.com"},"Action":"sts:AssumeRole"}]}'
aws iam put-role-policy --role-name gene-web-tier2-self-stop \
  --policy-name self-stop --policy-document file://infra/iam/tier2-self-stop-policy.json
aws iam create-instance-profile --instance-profile-name gene-web-tier2-self-stop
aws iam add-role-to-instance-profile --instance-profile-name gene-web-tier2-self-stop --role-name gene-web-tier2-self-stop

# attach the self-stop profile to the Tier 2 instance you just launched
aws ec2 associate-iam-instance-profile \
  --instance-id <TIER2_INSTANCE_ID> \
  --iam-instance-profile Name=gene-web-tier2-self-stop
```

Both policies are scoped so Tier 1 can only *start* the tagged Tier 2
instance and Tier 2 can only *stop itself* — neither can touch anything
else in your account. (EC2's `Describe*` actions don't support
resource-level restriction — that's an AWS limitation, not a scoping gap;
see the `_comment` in each policy file.)

### 3. Fill in Tier 2 secrets and apply manifests

SSH into Tier 2, edit `k8s/tier2/secrets.yaml` with a real `JWT_SECRET`
(must exactly match Tier 1's — see step 4), then apply the manifests:

```bash
sudo bash /opt/gene-web/infra/tier2/apply-tier2.sh
```

`database-credentials` is deliberately **not** in `secrets.yaml` at all —
apply it directly, every time, and never add it back to that file (a
prior version had it there, and re-applying the file after setting the
real value silently reset it to a placeholder in production):

```bash
kubectl create secret generic database-credentials -n gene-web \
  --from-literal=DATABASE_URL='<your real Neon/Postgres URL>' \
  --dry-run=client -o yaml | kubectl apply -f -
```

### 4. Launch Tier 1

- AMI: Ubuntu 22.04 LTS, `t4g.small`, 20GB gp3
- IAM instance profile: `gene-web-tier1-wake`
- User data: [`infra/tier1/userdata.sh`](tier1/userdata.sh) (it will stop and
  ask you to create `.env` on first run — see below — then re-run it)
- Security group: 80/443 open, 22 restricted to your IP

Create `/opt/gene-web/.env` with (in addition to your existing app vars —
`DATABASE_URL`, `JWT_SECRET`, `SALT_ROUNDS`, etc.):

```
FE_PORT=3000
AUTH_PORT=4000
WAKE_GATEWAY_PORT=4100
WAKE_SHARED_SECRET=<openssl rand -hex 32>
AWS_REGION=us-east-1
TIER2_INSTANCE_ID=<from step 1>
TIER2_HEALTH_URL=http://api.<TIER2_PUBLIC_IP>.nip.io/models/docs
```

Also add `NEXT_PUBLIC_WAKE_SHARED_SECRET=<same value as WAKE_SHARED_SECRET>`
to `apps/web/.env.production` — Next.js inlines `NEXT_PUBLIC_*` vars at
**build** time, so it must be set before `npm run build` runs (the user-data
script builds after you've created `.env`, so just make sure both files
exist before re-running it).

Then re-run: `sudo bash /opt/gene-web/infra/tier1/userdata.sh`

## Files

- `apps/wake_gateway/` — the always-on control service (`POST /wake`, `GET /wake/status`)
- `apps/web/hooks/useTier2Wake.ts` + `apps/web/app/dashboard/components/Tier2WakeGate.tsx` — frontend integration; wrap any ML feature in `<Tier2WakeGate>`
- `k8s/tier2/` — trimmed single-node manifests (replicas=1, right-sized requests) — this is what actually gets applied. `k8s/` and `k8s/local/` are left as-is as the original multi-replica EKS-target reference.
- `infra/iam/` — the two least-privilege policies
- `infra/tier1/`, `infra/tier2/` — user-data, PM2 config, nginx config, idle-checker

## Adding your other 2-3 projects

- Lightweight frontend/API → another PM2 app block in `infra/tier1/ecosystem.config.cjs` + another `location` in the nginx template, same box.
- Heavy compute → its own namespace in the same Tier 2 cluster (`kubectl apply -f` its own manifests there), so it rides the same wake/sleep switch instead of needing its own always-billing instance.
