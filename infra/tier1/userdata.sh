#!/bin/bash
# EC2 user-data for Tier 1 (the small always-on box). Installs Node, PM2,
# Nginx, clones the repo, and gets frontend + auth + wake-gateway running.
# Does NOT write secrets - create /opt/gene-web/.env yourself after this
# runs (see infra/README.md), then run this script's second half manually
# (or just re-run it; it's idempotent) to build and start the apps.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive

APP_ROOT="/opt/gene-web"
REPO_URL="https://github.com/SmartSood/BioInformatics_gene_expression.git"

apt-get update
apt-get install -y curl git nginx gettext-base

if ! command -v node >/dev/null 2>&1; then
  curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
  apt-get install -y nodejs
fi
npm install -g pm2

mkdir -p "${APP_ROOT}"
if [[ ! -d "${APP_ROOT}/.git" ]]; then
  git clone "${REPO_URL}" "${APP_ROOT}"
fi
cd "${APP_ROOT}"

if [[ ! -f .env ]]; then
  echo "!! ${APP_ROOT}/.env is missing. Create it (see infra/README.md), then re-run this script. Stopping here."
  exit 1
fi
set -a
source .env
set +a

npm install
npm run build --workspace apps/web --workspace apps/auth_backend --workspace apps/wake_gateway

FE_PORT="${FE_PORT}" AUTH_PORT="${AUTH_PORT}" WAKE_GATEWAY_PORT="${WAKE_GATEWAY_PORT:-4100}" \
  envsubst '${FE_PORT} ${AUTH_PORT} ${WAKE_GATEWAY_PORT}' \
  < infra/tier1/nginx.conf.template > /etc/nginx/sites-available/gene-web
ln -sf /etc/nginx/sites-available/gene-web /etc/nginx/sites-enabled/gene-web
rm -f /etc/nginx/sites-enabled/default
nginx -t && systemctl reload nginx || systemctl restart nginx

pm2 start infra/tier1/ecosystem.config.cjs
pm2 save
pm2 startup systemd -u root --hp /root | tail -1 | bash || true

echo "Tier 1 up. Frontend on :80, auth proxied at /auth, wake-gateway at /wake"
