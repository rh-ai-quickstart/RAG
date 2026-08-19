#!/bin/bash
# ============================================================================
# RAG Installer — Build Script
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Configuration
CONTAINER_ENGINE="${CONTAINER_ENGINE:-podman}"
REGISTRY="quay.io/rh-ai-quickstart"
IMAGE_NAME="rag-installer"
VERSION="0.2.38"
FULL_IMAGE="${REGISTRY}/${IMAGE_NAME}:${VERSION}"

GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

info() { echo -e "${GREEN}✓${NC} $1"; }
error() { echo -e "${RED}✗${NC} $1"; exit 1; }

cd "$PROJECT_DIR"

# Verify required files exist
info "Checking required files..."
REQUIRED_FILES=(
  "installer/entrypoint.sh"
  "installer/lib/install.sh"
  "installer/lib/uninstall.sh"
  "installer/lib/status.sh"
  "installer/lib/check_pre_reqs.sh"
  "installer/lib/upgrade.sh"
  "installer/Dockerfile"
  "deploy/helm/rag/Chart.yaml"
  "deploy/helm/rag/values.yaml"
  "deploy/helm/rag-values.yaml.example"
)

for file in "${REQUIRED_FILES[@]}"; do
  if [[ ! -e "$file" ]]; then
    error "Required file missing: $file"
  fi
done

# Build Helm chart dependencies
info "Building Helm chart dependencies..."
helm dependency update deploy/helm/rag 2>/dev/null || true
helm dependency build deploy/helm/rag

# Build the image
info "Building installer image: ${FULL_IMAGE}"
${CONTAINER_ENGINE} build --platform linux/amd64 -t "${FULL_IMAGE}" -f installer/Dockerfile .

info "Tagging as latest: ${REGISTRY}/${IMAGE_NAME}:latest"
${CONTAINER_ENGINE} tag "${FULL_IMAGE}" "${REGISTRY}/${IMAGE_NAME}:latest"

info "Build complete!"
echo ""
echo "Image: ${FULL_IMAGE}"
echo "Also tagged: ${REGISTRY}/${IMAGE_NAME}:latest"

# Push if requested
if [[ "${1:-}" == "push" ]]; then
  echo ""
  info "Pushing to registry..."
  ${CONTAINER_ENGINE} push "${FULL_IMAGE}"
  ${CONTAINER_ENGINE} push "${REGISTRY}/${IMAGE_NAME}:latest"
  info "Push complete!"
  echo ""
  echo "Image pushed to registry!"
  echo ""
  echo "Deploy to cluster:"
  echo "  ./installer/deploy.sh check_pre_reqs <namespace> - Validate prerequisites"
  echo "  ./installer/deploy.sh status <namespace>         - Check deployment status"
  echo "  ./installer/deploy.sh install <namespace>        - Deploy installation"
fi
