from __future__ import annotations

import json
import os
import time
from typing import Dict, Tuple

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.247, 0.243, 0.261)

CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

_preprocess = transforms.Compose(
    [
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ]
)


def preprocess_image(img: Image.Image) -> torch.Tensor:
    img = img.convert("RGB")
    x = _preprocess(img)
    return x.unsqueeze(0)  # (1, C, H, W)


def softmax_probs(logits: torch.Tensor) -> np.ndarray:
    probs = torch.softmax(logits, dim=1).detach().cpu().numpy()[0]
    return probs


def extract_drift_features(img: Image.Image) -> Dict[str, float]:
    """Lightweight numeric features for drift monitoring."""
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    # Channel means & stds
    means = arr.mean(axis=(0, 1))
    stds = arr.std(axis=(0, 1))
    # Overall brightness/contrast
    brightness = float(arr.mean())
    contrast = float(arr.std())

    return {
        "mean_r": float(means[0]),
        "mean_g": float(means[1]),
        "mean_b": float(means[2]),
        "std_r": float(stds[0]),
        "std_g": float(stds[1]),
        "std_b": float(stds[2]),
        "brightness": brightness,
        "contrast": contrast,
    }


def append_inference_log(features: Dict[str, float], pred: int, out_path: str | None = None) -> None:
    out_path = out_path or os.getenv("INFERENCE_LOG_PATH", "./data/inference_logs.jsonl")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    record = {
        "ts": time.time(),
        "pred": int(pred),
        **features,
    }

    with open(out_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
