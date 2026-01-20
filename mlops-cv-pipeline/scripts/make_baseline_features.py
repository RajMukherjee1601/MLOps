#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import time

from torchvision import datasets

from src.serving.utils import extract_drift_features


def parse_args():
    p = argparse.ArgumentParser(description="Create baseline feature jsonl from CIFAR-10")
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--split", choices=["train", "test"], default="train")
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--out", default="./data/baseline_features.jsonl")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    ds = datasets.CIFAR10(root=args.data_dir, train=(args.split == "train"), download=True)
    n = min(args.n, len(ds))

    with open(args.out, "w", encoding="utf-8") as f:
        for i in range(n):
            img, _ = ds[i]  # PIL image
            feats = extract_drift_features(img)
            rec = {"ts": time.time(), "pred": -1, **feats}
            f.write(json.dumps(rec) + "\n")

    print(f"Wrote baseline features: {args.out} ({n} rows)")


if __name__ == "__main__":
    main()
