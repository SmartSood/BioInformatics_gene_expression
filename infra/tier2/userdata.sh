#!/bin/bash
# EC2 user-data for Tier 2 (the on-demand k3s node). Runs once on first
# boot. Installs k3s + aws-cli + the idle self-stop timer, and clones the
# repo. Deliberately does NOT `kubectl apply` anything or touch secrets -
# user-data is visible to anyone with ec2:DescribeInstanceAttribute on this
# instance, so it must never carry real credentials. Apply manifests over
# SSH after filling in k8s/tier2/secrets.yaml for real (see infra/README.md).
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive

K3S_CHANNEL="v1.30"  # channel name, not a release tag - passed via INSTALL_K3S_CHANNEL below
APP_ROOT="/opt/gene-web"
REPO_URL="https://github.com/SmartSood/BioInformatics_gene_expression.git"

apt-get update
apt-get install -y curl git jq ca-certificates gnupg unzip

# AWS CLI (needed by idle-checker.sh to self-stop)
if ! command -v aws >/dev/null 2>&1; then
  curl -fsSL "https://awscli.amazonaws.com/awscli-exe-linux-$(uname -m).zip" -o /tmp/awscliv2.zip
  unzip -q /tmp/awscliv2.zip -d /tmp
  /tmp/aws/install
fi

if ! command -v docker >/dev/null 2>&1; then
  apt-get install -y docker.io
fi
systemctl enable --now docker

curl -sfL https://get.k3s.io | INSTALL_K3S_CHANNEL="${K3S_CHANNEL}" sh -
export KUBECONFIG=/etc/rancher/k3s/k3s.yaml
kubectl wait --for=condition=Ready node --all --timeout=180s || true

mkdir -p "${APP_ROOT}"
if [[ ! -d "${APP_ROOT}/.git" ]]; then
  git clone "${REPO_URL}" "${APP_ROOT}"
fi

# k3s auto-applies anything dropped in server/manifests/
mkdir -p /var/lib/rancher/k3s/server/manifests
cp "${APP_ROOT}/infra/tier2/traefik-access-log.yaml" /var/lib/rancher/k3s/server/manifests/

# Idle self-stop timer
chmod +x "${APP_ROOT}/infra/tier2/idle-checker.sh"
cp "${APP_ROOT}/infra/tier2/idle-checker.service" /etc/systemd/system/idle-checker.service
cp "${APP_ROOT}/infra/tier2/idle-checker.timer" /etc/systemd/system/idle-checker.timer
systemctl daemon-reload
systemctl enable --now idle-checker.timer

echo "Tier 2 bootstrap complete. SSH in, fill k8s/tier2/secrets.yaml, then run infra/tier2/apply-tier2.sh"
