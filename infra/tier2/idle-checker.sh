#!/bin/bash
# Runs every IDLE_CHECK_INTERVAL (via systemd timer) on Tier 2 itself.
# Stops the instance once no real ingress traffic has been seen for
# IDLE_THRESHOLD_MINUTES. Health/readiness probes never reach Traefik
# (kubelet hits each pod directly), so any line in the Traefik access log
# is genuine external traffic.
set -euo pipefail

IDLE_THRESHOLD_MINUTES="${IDLE_THRESHOLD_MINUTES:-20}"
export KUBECONFIG=/etc/rancher/k3s/k3s.yaml

log() { echo "[idle-checker] $(date -u +%FT%TZ) $*"; }

TRAEFIK_POD="$(kubectl get pods -n kube-system -l app.kubernetes.io/name=traefik \
  -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)"

if [[ -z "${TRAEFIK_POD}" ]]; then
  log "traefik pod not found yet, skipping this cycle"
  exit 0
fi

SINCE="${IDLE_THRESHOLD_MINUTES}m"
RECENT_REQUESTS="$(kubectl logs -n kube-system "${TRAEFIK_POD}" --since="${SINCE}" 2>/dev/null | grep -c '"RequestPath"' || true)"

if [[ "${RECENT_REQUESTS}" -gt 0 ]]; then
  log "saw ${RECENT_REQUESTS} request(s) in the last ${IDLE_THRESHOLD_MINUTES}m, staying up"
  exit 0
fi

# No traffic in the window. Guard against stopping right after boot, before
# the log even has ${IDLE_THRESHOLD_MINUTES}m of history to judge from.
UPTIME_MINUTES="$(( $(cut -d. -f1 /proc/uptime) / 60 ))"
if [[ "${UPTIME_MINUTES}" -lt "${IDLE_THRESHOLD_MINUTES}" ]]; then
  log "instance only up ${UPTIME_MINUTES}m (< ${IDLE_THRESHOLD_MINUTES}m threshold), staying up"
  exit 0
fi

log "idle for ${IDLE_THRESHOLD_MINUTES}m+, self-stopping"

TOKEN="$(curl -fsS -X PUT "http://169.254.169.254/latest/api/token" \
  -H "X-aws-ec2-metadata-token-ttl-seconds: 60")"
INSTANCE_ID="$(curl -fsS -H "X-aws-ec2-metadata-token: ${TOKEN}" \
  http://169.254.169.254/latest/meta-data/instance-id)"
REGION="$(curl -fsS -H "X-aws-ec2-metadata-token: ${TOKEN}" \
  http://169.254.169.254/latest/meta-data/placement/region)"

aws ec2 stop-instances --region "${REGION}" --instance-ids "${INSTANCE_ID}"
