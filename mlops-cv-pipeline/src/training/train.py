from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, Tuple

import mlflow
import mlflow.pytorch
import torch
from mlflow.tracking import MlflowClient
from torch import nn
from torch.optim import Adam

from .data import CIFAR10_CLASSES, DataConfig, get_dataloaders
from .model import SimpleCifarCNN


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    return (preds == targets).float().mean().item()


def run_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: Adam | None,
    device: torch.device,
) -> Tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            logits = model(x)
            loss = criterion(logits, y)
            acc = accuracy(logits, y)

            if is_train:
                loss.backward()
                optimizer.step()

        total_loss += loss.item()
        total_acc += acc
        n_batches += 1

    return total_loss / max(1, n_batches), total_acc / max(1, n_batches)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train CIFAR-10 CNN with MLflow tracking")
    p.add_argument("--data-dir", default=os.getenv("DATA_DIR", "./data"))
    p.add_argument("--epochs", type=int, default=int(os.getenv("EPOCHS", "1")))
    p.add_argument("--batch-size", type=int, default=int(os.getenv("BATCH_SIZE", "64")))
    p.add_argument("--lr", type=float, default=float(os.getenv("LR", "0.001")))
    p.add_argument("--device", default=os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))

    p.add_argument("--tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    p.add_argument("--experiment", default=os.getenv("MLFLOW_EXPERIMENT_NAME", "cv-cifar10"))
    p.add_argument("--model-name", default=os.getenv("MODEL_NAME", "cifar10_cnn"))
    p.add_argument("--register", action="store_true", help="Register model to MLflow Model Registry")
    p.add_argument(
        "--stage",
        default=os.getenv("MODEL_STAGE", "Staging"),
        help="Stage to transition newest registered model to (e.g., Staging/Production)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)

    device = torch.device(args.device)

    data_cfg = DataConfig(data_dir=args.data_dir, batch_size=args.batch_size)
    train_loader, test_loader = get_dataloaders(data_cfg)

    model = SimpleCifarCNN(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.lr)

    # Store label mapping as an artifact
    os.makedirs("./reports", exist_ok=True)
    labels_path = "./reports/cifar10_classes.json"
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(CIFAR10_CLASSES, f, indent=2)

    with mlflow.start_run() as run:
        mlflow.log_params(
            {
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "device": str(device),
            }
        )
        mlflow.log_artifact(labels_path, artifact_path="metadata")

        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = run_epoch(model, test_loader, criterion, None, device)
            dt = time.time() - t0

            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "epoch_time_s": dt,
                },
                step=epoch,
            )

            print(
                f"Epoch {epoch}/{args.epochs} | "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
                f"time={dt:.1f}s"
            )

        # Log model
        # If register is True, MLflow will create/append a model version under args.model_name.
        mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path="model",
            registered_model_name=args.model_name if args.register else None,
        )

        print(f"MLflow run_id: {run.info.run_id}")

    # Transition the newest version to the requested stage
    if args.register:
        client = MlflowClient()
        versions = client.search_model_versions(f"name='{args.model_name}'")
        if not versions:
            raise RuntimeError("Model registration failed: no model versions found.")

        newest = max(versions, key=lambda v: int(v.version))
        client.transition_model_version_stage(
            name=args.model_name,
            version=newest.version,
            stage=args.stage,
            archive_existing_versions=False,
        )
        print(f"Registered '{args.model_name}' version {newest.version} -> stage {args.stage}")


if __name__ == "__main__":
    main()
