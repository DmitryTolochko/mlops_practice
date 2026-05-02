#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
SHORT_SHA=$(git rev-parse --short HEAD)
BRANCH=$(git rev-parse --abbrev-ref HEAD | tr '/' '-')
NAME="mlops-lab3-api"
TAG="${BRANCH}-${SHORT_SHA}"

docker build -t "${NAME}:${TAG}" -t "${NAME}:latest" .

if [[ -n "${DOCKERHUB_USER:-}" ]]; then
  docker tag "${NAME}:${TAG}" "${DOCKERHUB_USER}/${NAME}:${TAG}"
  docker tag "${NAME}:latest" "${DOCKERHUB_USER}/${NAME}:latest"
  echo "Tagged ${DOCKERHUB_USER}/${NAME}:${TAG}"
  if [[ "${1:-}" == "--push" ]]; then
    docker push "${DOCKERHUB_USER}/${NAME}:${TAG}"
    docker push "${DOCKERHUB_USER}/${NAME}:latest"
  fi
fi
