# Kubernetes Deployment Guide for Gene Web Platform

This directory contains Kubernetes manifests for deploying the Gene Web Platform on AWS EKS.

## Prerequisites

1. **AWS Account** with EKS cluster created
2. **kubectl** configured to access your EKS cluster
3. **Helm** (optional, for cert-manager and ingress-nginx)
4. **Docker Hub** or other container registry account
5. **AWS ECR** (Elastic Container Registry) for private image storage

## Architecture Overview

- **Namespace**: `gene-web` - isolated namespace for all resources
- **Backend Services**:
  - Model Backend (FastAPI) - port 8000
  - Embedding Backend (FastAPI) - port 8002
  - DepMap Backend (FastAPI) - port 8001
  - Affinity Backend (FastAPI) - port 8003
  - Auth Backend (Node.js) - port 3001
- **Frontend**: Web Frontend (Next.js) - port 3000
- **Storage**: AWS S3 for all artifacts (no local persistence)
- **Cache**: AWS ElastiCache (Redis)
- **Database**: AWS RDS (PostgreSQL) for Prisma

## File Structure

```
k8s/
├── namespace.yaml           # Kubernetes namespace
├── configmap.yaml           # Configuration for all services
├── secrets.yaml             # Secrets (AWS creds, JWT, DB URL)
├── model-backend.yaml       # Model training backend
├── embedding-backend.yaml   # Embedding service
├── depmap-backend.yaml      # DepMap association service
├── affinity-backend.yaml    # Affinity prediction service
├── frontend-auth.yaml       # Web frontend + auth backend
├── ingress.yaml             # Ingress routing + TLS
└── README.md                # This file
```

## Pre-Deployment Steps

### 1. Create Docker Images and Push to Registry

```bash
# Login to Docker Hub (or your registry)
docker login

# Build and tag images
docker build -f apps/model_backend/Dockerfile -t docker.io/your-username/gene-model:latest .
docker build -f apps/embedding_backend/Dockerfile -t docker.io/your-username/gene-embedding:latest .
docker build -f apps/depmap_backend/Dockerfile -t docker.io/your-username/gene-depmap:latest .
docker build -f apps/affinity_backend/Dockerfile -t docker.io/your-username/gene-affinity:latest .
docker build -f apps/auth_backend/Dockerfile -t docker.io/your-username/gene-auth:latest .
docker build -f apps/web/Dockerfile -t docker.io/your-username/gene-web:latest .

# Push images
docker push docker.io/your-username/gene-model:latest
docker push docker.io/your-username/gene-embedding:latest
docker push docker.io/your-username/gene-depmap:latest
docker push docker.io/your-username/gene-affinity:latest
docker push docker.io/your-username/gene-auth:latest
docker push docker.io/your-username/gene-web:latest
```

### 2. Update Image References

Edit all `*.yaml` files and replace:
- `docker-registry.example.com` with your actual Docker registry
- `your-username` with your Docker Hub username

### 3. Setup AWS Resources

#### Create RDS PostgreSQL Database
```bash
aws rds create-db-instance \
  --db-instance-identifier gene-web-db \
  --db-instance-class db.t3.micro \
  --engine postgres \
  --allocated-storage 20 \
  --master-username postgres \
  --master-user-password YOUR_PASSWORD
```

#### Create ElastiCache Redis Cluster
```bash
aws elasticache create-cache-cluster \
  --cache-cluster-id gene-web-redis \
  --cache-node-type cache.t3.micro \
  --engine redis \
  --num-cache-nodes 1
```

#### Create S3 Bucket
```bash
aws s3 mb s3://gene-web-data --region us-east-1
```

#### Create IAM Role for EKS Pods
```bash
# Create policy for S3 and RDS access
aws iam create-role --role-name gene-web-pod-role \
  --assume-role-policy-document file://trust-policy.json

# Attach S3 policy
aws iam attach-role-policy --role-name gene-web-pod-role \
  --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess
```

### 4. Update Secrets

Edit `secrets.yaml` with your actual values:
```yaml
S3_BUCKET: gene-web-data
S3_ACCESS_KEY: <your-aws-access-key>
S3_SECRET_KEY: <your-aws-secret-key>
DATABASE_URL: postgresql://postgres:PASSWORD@rds-endpoint:5432/gene_web
JWT_SECRET: <generate-strong-secret>
REDIS_URL: redis://redis-endpoint:6379/0
```

## Deployment

### 1. Create Namespace and ConfigMaps
```bash
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
```

### 2. Create Secrets
```bash
kubectl apply -f k8s/secrets.yaml
```

### 3. Deploy Services
```bash
# Deploy all backends
kubectl apply -f k8s/model-backend.yaml
kubectl apply -f k8s/embedding-backend.yaml
kubectl apply -f k8s/depmap-backend.yaml
kubectl apply -f k8s/affinity-backend.yaml
kubectl apply -f k8s/frontend-auth.yaml

# Wait for deployments to be ready
kubectl wait --for=condition=available --timeout=300s deployment --all -n gene-web
```

### 4. Setup Ingress

#### Install NGINX Ingress Controller (if not already installed)
```bash
helm repo add ingress-nginx https://kubernetes.github.io/ingress-nginx
helm repo update
helm install nginx ingress-nginx/ingress-nginx --namespace ingress-nginx --create-namespace
```

#### Install Cert-Manager (for Let's Encrypt TLS)
```bash
helm repo add jetstack https://charts.jetstack.io
helm repo update
helm install cert-manager jetstack/cert-manager \
  --namespace cert-manager --create-namespace \
  --set installCRDs=true
```

#### Deploy Ingress
```bash
# Update domain names in ingress.yaml first
kubectl apply -f k8s/ingress.yaml
```

### 5. Verify Deployment
```bash
# Check pod status
kubectl get pods -n gene-web

# Check services
kubectl get svc -n gene-web

# Check ingress
kubectl get ingress -n gene-web

# View pod logs
kubectl logs -n gene-web -l app=model-backend --tail=50
```

## Configuration Management

### Update ConfigMap
```bash
kubectl edit configmap shared-config -n gene-web
```

### Update Secrets
```bash
# Create a new secret version
kubectl create secret generic aws-credentials \
  --from-literal=S3_BUCKET=new-bucket \
  --from-literal=S3_ACCESS_KEY=new-key \
  --from-literal=S3_SECRET_KEY=new-secret \
  -n gene-web --dry-run=client -o yaml | kubectl apply -f -

# Restart deployments to pick up new secrets
kubectl rollout restart deployment -n gene-web
```

## Scaling

### Manual Scaling
```bash
# Scale model backend to 5 replicas
kubectl scale deployment model-backend-api -n gene-web --replicas=5
```

### Auto-Scaling (HPA)
HPA is configured in the manifest files. Monitor with:
```bash
kubectl get hpa -n gene-web -w
```

## Monitoring and Troubleshooting

### Check Pod Status
```bash
kubectl describe pod <pod-name> -n gene-web
```

### Stream Logs
```bash
kubectl logs -f deployment/model-backend-api -n gene-web
```

### Port Forward for Local Testing
```bash
kubectl port-forward svc/model-backend 8000:8000 -n gene-web
```

### Check Resource Usage
```bash
kubectl top pod -n gene-web
kubectl top node
```

## CI/CD Integration

For automated deployments:
1. Push Docker images to registry on each commit
2. Update deployment image tags in manifests
3. Apply updated manifests to EKS cluster

See `.github/workflows/deploy.yaml` for GitHub Actions workflow.

## Cleanup

To remove all resources:
```bash
kubectl delete namespace gene-web
```

This will delete all deployments, services, and resources in the namespace.

## Important Notes

1. **Secrets in Git**: Never commit actual secrets to git. Use external secret management tools like:
   - AWS Secrets Manager
   - HashiCorp Vault
   - Sealed Secrets

2. **Image Registry**: Update all image references to point to your actual registry

3. **Domain Names**: Replace `example.com` with your actual domain

4. **Resource Limits**: Adjust CPU and memory requests/limits based on your workload

5. **Database Migrations**: Run Prisma migrations before deploying:
   ```bash
   kubectl exec -it deployment/model-backend-api -n gene-web -- \
     npx prisma migrate deploy
   ```
