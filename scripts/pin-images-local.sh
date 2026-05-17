#!/usr/bin/env bash
set -euo pipefail

if [ $# -ne 1 ]; then
  echo "Usage: $0 <image-tag>"
  exit 1
fi
IMAGE_TAG=$1

echo "Pinning k8s/local manifests to image tag: ${IMAGE_TAG}"
for f in k8s/local/*.yaml; do
  echo "  -> $f"
  sed "s/:IMAGE_TAG/:${IMAGE_TAG}/g" "$f" > "$f.tmp" && mv "$f.tmp" "$f"
done

echo "Done. Commit the changes if you want them persisted: git add k8s/local/*.yaml && git commit -m 'pin images to ${IMAGE_TAG}'" 
