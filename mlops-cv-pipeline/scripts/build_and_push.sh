#!/usr/bin/env bash
set -euo pipefail

# Example:
#   export IMAGE=ghcr.io/<user-or-org>/cv-inference:latest
#   echo $GHCR_TOKEN | docker login ghcr.io -u <user> --password-stdin
#   ./scripts/build_and_push.sh

: "${IMAGE:?Set IMAGE (e.g., ghcr.io/<user>/cv-inference:latest)}"

docker build -f docker/Dockerfile.serve -t "$IMAGE" .
docker push "$IMAGE"
