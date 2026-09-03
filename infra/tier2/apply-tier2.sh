#!/bin/bash
# Run this over SSH on Tier 2. k8s/tier2/secrets.yaml only carries the
# jwt-secret placeholder (fill in the real value to match Tier 1's
# JWT_SECRET after this runs). database-credentials is NOT in that file at
# all - apply it separately, every time, via `kubectl create secret
# --dry-run=client -o yaml | kubectl apply -f -` (see infra/README.md).
# Re-running this script does NOT touch DATABASE_URL, by design - a prior
# version had it in secrets.yaml and re-applying that file silently
# clobbered the real value back to a placeholder in production.
set -euo pipefail

ROOT="${1:-/opt/gene-web}"
export KUBECONFIG=/etc/rancher/k3s/k3s.yaml

PUBLIC_IP="$(curl -fsS http://169.254.169.254/latest/meta-data/public-ipv4 || true)"
if [[ -z "${PUBLIC_IP}" ]]; then
  PUBLIC_IP="127.0.0.1"
fi
sed -i "s/EC2_PUBLIC_IP/${PUBLIC_IP}/g" "${ROOT}/k8s/tier2/ingress.yaml"

kubectl apply -f "${ROOT}/k8s/tier2/namespace.yaml"
kubectl apply -f "${ROOT}/k8s/tier2/configmap.yaml"
kubectl apply -f "${ROOT}/k8s/tier2/secrets.yaml"
kubectl apply -f "${ROOT}/k8s/tier2/redis.yaml"
kubectl apply -f "${ROOT}/k8s/tier2/postgres.yaml"
kubectl apply -f "${ROOT}/k8s/tier2/model-backend.yaml"
kubectl apply -f "${ROOT}/k8s/tier2/embedding-backend.yaml"
kubectl apply -f "${ROOT}/k8s/tier2/depmap-backend.yaml"
kubectl apply -f "${ROOT}/k8s/tier2/affinity-backend.yaml"
kubectl apply -f "${ROOT}/k8s/tier2/ingress.yaml"

echo "Applied. Tier 2 API reachable at http://api.${PUBLIC_IP}.nip.io once pods are Ready:"
echo "  kubectl get pods -n gene-web -w"
