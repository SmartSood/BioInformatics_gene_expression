# Single-EC2 k3s Stack

This directory contains a local Kubernetes variant for running the platform on one EC2 instance.

## What changes from EKS

- Redis runs in-cluster as `redis-elasticache` and is ephemeral.
- PostgreSQL runs in-cluster as a StatefulSet with `local-path` storage.
- S3 remains the artifact store, but EC2 should use an IAM role instead of static access keys.
- Ingress uses k3s's default `traefik` controller and `nip.io` hostnames.

## Bootstrap

Use `scripts/bootstrap-ec2-userdata.sh` as EC2 user-data or run it manually on a fresh Ubuntu instance.

## Apply order

1. `kubectl apply -f k8s/local/namespace.yaml`
2. `kubectl apply -f k8s/local/configmap.yaml`
3. `kubectl apply -f k8s/local/secrets.yaml`
4. `kubectl apply -f k8s/local/redis.yaml`
5. `kubectl apply -f k8s/local/postgres.yaml`
6. `kubectl apply -f k8s/local/model-backend.yaml`
7. `kubectl apply -f k8s/local/embedding-backend.yaml`
8. `kubectl apply -f k8s/local/depmap-backend.yaml`
9. `kubectl apply -f k8s/local/affinity-backend.yaml`
10. `kubectl apply -f k8s/local/frontend-auth.yaml`
11. `kubectl apply -f k8s/local/ingress.yaml`

## Notes

- The frontend hostnames are set to `*.127.0.0.1.nip.io` so they can be changed easily after the EC2 public IP is known.
- If you prefer a real domain, replace those hostnames in `k8s/local/ingress.yaml`.