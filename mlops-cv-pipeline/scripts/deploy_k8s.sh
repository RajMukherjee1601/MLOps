#!/usr/bin/env bash
set -euo pipefail

# Requires kubectl configured.

kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/
