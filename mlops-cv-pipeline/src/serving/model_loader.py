from __future__ import annotations

import os
from functools import lru_cache

import mlflow
import mlflow.pytorch
import torch


@lru_cache(maxsize=1)
def load_model() -> torch.nn.Module:
    """Load a PyTorch model from MLflow Model Registry."""
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    model_name = os.getenv("MODEL_NAME", "cifar10_cnn")
    stage = os.getenv("MODEL_STAGE", "Staging")

    mlflow.set_tracking_uri(tracking_uri)

    model_uri = f"models:/{model_name}/{stage}"
    model = mlflow.pytorch.load_model(model_uri)
    model.eval()
    return model
