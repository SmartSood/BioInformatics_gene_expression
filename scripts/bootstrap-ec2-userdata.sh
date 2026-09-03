#!/bin/bash
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive

K3S_CHANNEL="v1.30"
K8S_NAMESPACE="gene-web"
APP_ROOT="/opt/gene-web"
LOCAL_MANIFEST_DIR="${APP_ROOT}/k8s/local"

USE_GPU="${USE_GPU:-false}"
INSTALL_GITHUB_RUNNER="${INSTALL_GITHUB_RUNNER:-false}"
GITHUB_OWNER="${GITHUB_OWNER:-}"
GITHUB_REPO="${GITHUB_REPO:-}"
GITHUB_RUNNER_TOKEN="${GITHUB_RUNNER_TOKEN:-}"

apt-get update
apt-get install -y curl git jq ca-certificates gnupg lsb-release unzip tar

if ! command -v docker >/dev/null 2>&1; then
  apt-get install -y docker.io
fi
systemctl enable --now docker

if [[ "${USE_GPU}" == "true" ]]; then
  apt-get install -y linux-headers-$(uname -r) build-essential
  apt-get install -y ubuntu-drivers-common
  if command -v ubuntu-drivers >/dev/null 2>&1; then
    ubuntu-drivers autoinstall || true
  fi

  distribution="$(. /etc/os-release; echo "${ID}${VERSION_ID}")"
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit.gpg
  curl -fsSL "https://nvidia.github.io/libnvidia-container/${distribution}/libnvidia-container.list" \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit.gpg] https://#g' \
    > /etc/apt/sources.list.d/nvidia-container-toolkit.list
  apt-get update
  apt-get install -y nvidia-container-toolkit
  nvidia-ctk runtime configure --runtime=docker
  systemctl restart docker
fi

curl -sfL https://get.k3s.io | INSTALL_K3S_CHANNEL="${K3S_CHANNEL}" sh -

export KUBECONFIG=/etc/rancher/k3s/k3s.yaml
mkdir -p "${APP_ROOT}"

kubectl wait --for=condition=Ready node --all --timeout=180s || true

mkdir -p /var/lib/rancher/k3s/server/manifests

if [[ "${INSTALL_GITHUB_RUNNER}" == "true" && -n "${GITHUB_OWNER}" && -n "${GITHUB_REPO}" && -n "${GITHUB_RUNNER_TOKEN}" ]]; then
  RUNNER_DIR="/opt/actions-runner"
  mkdir -p "${RUNNER_DIR}"
  cd "${RUNNER_DIR}"
  if [[ ! -f actions-runner-linux-x64.tar.gz ]]; then
    curl -fsSL -o actions-runner-linux-x64.tar.gz \
      https://github.com/actions/runner/releases/download/v2.323.0/actions-runner-linux-x64-2.323.0.tar.gz
    tar xzf actions-runner-linux-x64.tar.gz
  fi
  ./config.sh --unattended --url "https://github.com/${GITHUB_OWNER}/${GITHUB_REPO}" --token "${GITHUB_RUNNER_TOKEN}" --name "gene-web-ec2" --labels "self-hosted,ec2,k3s"
  ./svc.sh install
  ./svc.sh start
fi

cat >/usr/local/bin/k3s-load-image <<'SH'
#!/bin/bash
set -euo pipefail

IMAGE="${1:?image name required}"
TAG="${2:-latest}"

docker build -t "${IMAGE}:${TAG}" .
docker save "${IMAGE}:${TAG}" | k3s ctr images import -
SH
chmod +x /usr/local/bin/k3s-load-image

cat >/usr/local/bin/apply-gene-web-local <<'SH'
#!/bin/bash
set -euo pipefail

ROOT="${1:-/opt/gene-web}"
export KUBECONFIG=/etc/rancher/k3s/k3s.yaml

PUBLIC_IP="$(curl -fsS http://169.254.169.254/latest/meta-data/public-ipv4 || true)"
if [[ -z "${PUBLIC_IP}" ]]; then
  PUBLIC_IP="127.0.0.1"
fi

sed -i "s/EC2_PUBLIC_IP/${PUBLIC_IP}/g" "${ROOT}/k8s/local/configmap.yaml"
sed -i "s/EC2_PUBLIC_IP/${PUBLIC_IP}/g" "${ROOT}/k8s/local/ingress.yaml"

kubectl apply -f "${ROOT}/k8s/local/namespace.yaml"
kubectl apply -f "${ROOT}/k8s/local/configmap.yaml"
kubectl apply -f "${ROOT}/k8s/local/secrets.yaml"
kubectl apply -f "${ROOT}/k8s/local/redis.yaml"
kubectl apply -f "${ROOT}/k8s/local/postgres.yaml"
kubectl apply -f "${ROOT}/k8s/local/model-backend.yaml"
kubectl apply -f "${ROOT}/k8s/local/embedding-backend.yaml"
kubectl apply -f "${ROOT}/k8s/local/depmap-backend.yaml"
kubectl apply -f "${ROOT}/k8s/local/affinity-backend.yaml"
kubectl apply -f "${ROOT}/k8s/local/frontend-auth.yaml"
kubectl apply -f "${ROOT}/k8s/local/ingress.yaml"
SH
chmod +x /usr/local/bin/apply-gene-web-local

echo "Bootstrap complete. Clone the repo to ${APP_ROOT} and run /usr/local/bin/apply-gene-web-local after setting your ingress host values."