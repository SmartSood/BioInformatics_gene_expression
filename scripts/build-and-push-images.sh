#!/usr/bin/env bash
set -euo pipefail

IMAGE_TAG=$(git rev-parse --short HEAD)
DOCKER_USERNAME=${DOCKER_USERNAME:-smarthcnbl}
PUSH_IMAGES=${PUSH_IMAGES:-true}
LOAD_TO_K3S=${LOAD_TO_K3S:-false}

import_to_k3s() {
  local image_ref="$1"
  echo "  Loading ${image_ref} into k3s containerd..."
  docker save "${image_ref}" | sudo k3s ctr images import - > /tmp/$(echo "${image_ref}" | tr '/:' '_')-import.log 2>&1
}

echo "=========================================="
echo "Building images with tag: ${IMAGE_TAG}"
echo "Using username: ${DOCKER_USERNAME}"
echo "Push images: ${PUSH_IMAGES}"
echo "Load to k3s: ${LOAD_TO_K3S}"
echo "=========================================="
echo ""

# Step 1: Build shared base image first (torch, transformers, etc. built ONCE)
echo "▶ [BUILD BASE] shared-base:${IMAGE_TAG} (this may take 5-10 min)"
echo "  Installing torch, transformers, rdkit, scikit-learn, pandas... once"
build_start=$(date +%s)

if docker build -f apps/shared-base.Dockerfile -t "${DOCKER_USERNAME}/shared-base:${IMAGE_TAG}" . > /tmp/shared-base-build.log 2>&1; then
  build_end=$(date +%s)
  build_duration=$((build_end - build_start))
  echo "✓ [OK] shared-base built in ${build_duration}s"
else
  echo "✗ [FAILED] shared-base build failed"
  echo "  Last 50 lines of log:"
  tail -50 /tmp/shared-base-build.log | sed 's/^/  /'
  exit 1
fi

# Also tag as latest for convenience
docker tag "${DOCKER_USERNAME}/shared-base:${IMAGE_TAG}" "${DOCKER_USERNAME}/shared-base:latest"

echo ""
if [ "${PUSH_IMAGES}" = "true" ]; then
  echo "Pushing shared-base..."
  if docker push "${DOCKER_USERNAME}/shared-base:${IMAGE_TAG}" > /tmp/shared-base-push.log 2>&1; then
    echo "✓ [OK] shared-base pushed"
  else
    echo "✗ [FAILED] shared-base push failed"
    tail -20 /tmp/shared-base-push.log | sed 's/^/  /'
    exit 1
  fi
else
  echo "Skipping shared-base push (PUSH_IMAGES=false)"
fi

echo ""
echo "=========================================="
echo "Shared base ready! Now building services..."
echo "=========================================="
echo ""

# Step 2: Build all services (much faster now - no torch re-download)
services=(
  "model_backend:gene-model"
  "embedding_backend:gene-embedding"
  "depmap_backend:gene-depmap"
  "affinity_backend:gene-affinity"
  "auth_backend:gene-auth"
  "web:gene-web"
)

start_time=$(date +%s)
for service in "${services[@]}"; do
  path="${service%%:*}"
  image="${service##*:}"
  
  build_start=$(date +%s)
  echo "▶ [BUILD] ${image}:${IMAGE_TAG} (started at $(date +%H:%M:%S))"
  
  if docker build -f "apps/${path}/Dockerfile" -t "${DOCKER_USERNAME}/${image}:${IMAGE_TAG}" . > /tmp/${image}-build.log 2>&1; then
    build_end=$(date +%s)
    build_duration=$((build_end - build_start))
    echo "✓ [OK] ${image} built in ${build_duration}s (FAST - using shared-base)"
    if [ "${LOAD_TO_K3S}" = "true" ]; then
      import_to_k3s "${DOCKER_USERNAME}/${image}:${IMAGE_TAG}"
      echo "✓ [OK] ${image} loaded to k3s"
    fi
  else
    echo "✗ [FAILED] ${image} build failed"
    echo "  Last 30 lines of log:"
    tail -30 /tmp/${image}-build.log | sed 's/^/  /'
    exit 1
  fi
done

if [ "${PUSH_IMAGES}" = "true" ]; then
  echo ""
  echo "=========================================="
  echo "All builds complete! Starting pushes..."
  echo "=========================================="
  echo ""

  # Step 3: Push all services
  for service in "${services[@]}"; do
    image="${service##*:}"

    push_start=$(date +%s)
    echo "▶ [PUSH] ${DOCKER_USERNAME}/${image}:${IMAGE_TAG} (started at $(date +%H:%M:%S))"

    if docker push "${DOCKER_USERNAME}/${image}:${IMAGE_TAG}" > /tmp/${image}-push.log 2>&1; then
      push_end=$(date +%s)
      push_duration=$((push_end - push_start))
      echo "✓ [OK] ${image} pushed in ${push_duration}s"
    else
      echo "✗ [FAILED] ${image} push failed"
      echo "  Last 20 lines of log:"
      tail -20 /tmp/${image}-push.log | sed 's/^/  /'
      exit 1
    fi
  done
else
  echo ""
  echo "Skipping service pushes (PUSH_IMAGES=false)"
fi

end_time=$(date +%s)
total_duration=$((end_time - start_time))

echo ""
echo "=========================================="
echo "✓ SUCCESS!"
if [ "${PUSH_IMAGES}" = "true" ]; then
  echo "Shared base + all services built & pushed"
else
  echo "Shared base + all services built locally"
fi
if [ "${LOAD_TO_K3S}" = "true" ]; then
  echo "Images were loaded into k3s containerd"
fi
echo "Tag: ${IMAGE_TAG}"
echo "Total time: ${total_duration}s ($(( total_duration / 60 ))m $(( total_duration % 60 ))s)"
echo "=========================================="