#!/usr/bin/env bash
# scripts/export_docker.sh
# Exports the full stack as a .rar archive and pushes to Docker Hub.
# Usage: ./scripts/export_docker.sh <dockerhub-username>

set -e

USERNAME="${1:-your-dockerhub-username}"
TAG="legalrag:latest"
HUB_TAG="${USERNAME}/legalrag:latest"
EXPORT_FILE="legalrag_docker.rar"

echo "🐳 Building Docker images …"
docker compose build --no-cache

echo "📦 Saving image to tar then compressing …"
docker save legalrag-backend legalrag-frontend legalrag-triton 2>/dev/null || \
docker compose images -q | xargs docker save -o legalrag_stack.tar

echo "🗜️  Creating RAR archive …"
if command -v rar &>/dev/null; then
    rar a -m5 "${EXPORT_FILE}" legalrag_stack.tar
else
    echo "⚠️  'rar' not found. Installing via apt …"
    sudo apt-get install -y rar
    rar a -m5 "${EXPORT_FILE}" legalrag_stack.tar
fi

echo "☁️  Tagging and pushing to Docker Hub as ${HUB_TAG} …"
docker tag legalrag-backend:latest "${USERNAME}/legalrag-backend:latest"
docker tag legalrag-frontend:latest "${USERNAME}/legalrag-frontend:latest"
docker push "${USERNAME}/legalrag-backend:latest"
docker push "${USERNAME}/legalrag-frontend:latest"

echo ""
echo "✅ Done!"
echo "   Archive: ${EXPORT_FILE}  ($(du -sh ${EXPORT_FILE} | cut -f1))"
echo "   Docker Hub: https://hub.docker.com/r/${USERNAME}/legalrag-backend"
echo ""
echo "Pull command for others:"
echo "   docker pull ${USERNAME}/legalrag-backend:latest"
echo "   docker pull ${USERNAME}/legalrag-frontend:latest"
