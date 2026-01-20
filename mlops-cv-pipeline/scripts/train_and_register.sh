#!/usr/bin/env bash
set -euo pipefail

: "${MLFLOW_TRACKING_URI:=http://localhost:5000}"
: "${MLFLOW_EXPERIMENT_NAME:=cv-cifar10}"
: "${MODEL_NAME:=cifar10_cnn}"
: "${EPOCHS:=1}"
: "${BATCH_SIZE:=64}"

export MLFLOW_TRACKING_URI MLFLOW_EXPERIMENT_NAME MODEL_NAME

python -m src.training.train \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --register
