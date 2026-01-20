from __future__ import annotations

import argparse
import json
import math
import os
from typing import Dict, List, Tuple

import numpy as np


DEFAULT_FEATURES = [
    "mean_r",
    "mean_g",
    "mean_b",
    "std_r",
    "std_g",
    "std_b",
    "brightness",
    "contrast",
]


def read_jsonl(path: str) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def psi(baseline: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index (PSI). Higher = more drift.

    Rule-of-thumb (often used):
    - <0.1: little/no drift
    - 0.1-0.25: moderate drift
    - >0.25: large drift
    """
    # Use baseline quantiles for stable bin edges
    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.quantile(baseline, quantiles)
    # Ensure unique edges
    edges = np.unique(edges)
    if len(edges) < 3:
        return 0.0

    b_hist, _ = np.histogram(baseline, bins=edges)
    c_hist, _ = np.histogram(current, bins=edges)

    b_pct = b_hist / max(1, b_hist.sum())
    c_pct = c_hist / max(1, c_hist.sum())

    # Avoid zeros
    eps = 1e-6
    b_pct = np.clip(b_pct, eps, 1)
    c_pct = np.clip(c_pct, eps, 1)

    return float(np.sum((c_pct - b_pct) * np.log(c_pct / b_pct)))


def drift_summary(
    baseline_rows: List[Dict[str, float]],
    current_rows: List[Dict[str, float]],
    features: List[str],
) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for feat in features:
        b = np.array([r.get(feat, np.nan) for r in baseline_rows], dtype=np.float32)
        c = np.array([r.get(feat, np.nan) for r in current_rows], dtype=np.float32)
        b = b[~np.isnan(b)]
        c = c[~np.isnan(c)]
        if len(b) == 0 or len(c) == 0:
            continue

        out[feat] = {
            "psi": psi(b, c),
            "baseline_mean": float(np.mean(b)),
            "current_mean": float(np.mean(c)),
            "baseline_std": float(np.std(b)),
            "current_std": float(np.std(c)),
        }
    return out


def pred_distribution(rows: List[Dict[str, float]], n_classes: int = 10) -> np.ndarray:
    preds = [int(r.get("pred", -1)) for r in rows]
    preds = [p for p in preds if 0 <= p < n_classes]
    if not preds:
        return np.zeros(n_classes, dtype=np.float32)
    hist = np.bincount(preds, minlength=n_classes).astype(np.float32)
    return hist / hist.sum()


def l1_dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a - b).sum())


def render_html(summary: Dict[str, Dict[str, float]], pred_l1: float, out_path: str) -> None:
    rows = []
    for feat, s in sorted(summary.items(), key=lambda kv: kv[1]["psi"], reverse=True):
        rows.append(
            f"<tr><td>{feat}</td><td>{s['psi']:.4f}</td><td>{s['baseline_mean']:.4f}</td><td>{s['current_mean']:.4f}</td><td>{s['baseline_std']:.4f}</td><td>{s['current_std']:.4f}</td></tr>"
        )

    html = f"""<!doctype html>
<html>
<head>
  <meta charset='utf-8' />
  <title>Drift Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; }}
    .badge {{ display:inline-block; padding: 4px 8px; border-radius: 8px; background:#eee; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 16px; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; }}
    th {{ background: #f7f7f7; text-align: left; }}
  </style>
</head>
<body>
  <h1>Drift Report</h1>
  <p class='badge'>Prediction distribution L1 distance: <b>{pred_l1:.4f}</b></p>

  <h2>Feature drift (PSI)</h2>
  <p>Rule of thumb: PSI &lt; 0.1 (low), 0.1–0.25 (moderate), &gt; 0.25 (high)</p>

  <table>
    <thead>
      <tr>
        <th>Feature</th>
        <th>PSI</th>
        <th>Baseline Mean</th>
        <th>Current Mean</th>
        <th>Baseline Std</th>
        <th>Current Std</th>
      </tr>
    </thead>
    <tbody>
      {''.join(rows)}
    </tbody>
  </table>
</body>
</html>
"""

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate a simple drift report from jsonl logs")
    p.add_argument("--baseline", required=True, help="Baseline jsonl file")
    p.add_argument("--current", required=True, help="Current jsonl file")
    p.add_argument("--out", default="./reports/drift_report.html", help="Output HTML path")
    p.add_argument("--features", nargs="*", default=DEFAULT_FEATURES)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    baseline_rows = read_jsonl(args.baseline)
    current_rows = read_jsonl(args.current)

    summary = drift_summary(baseline_rows, current_rows, args.features)

    b_pred = pred_distribution(baseline_rows)
    c_pred = pred_distribution(current_rows)
    pred_l1 = l1_dist(b_pred, c_pred)

    render_html(summary, pred_l1, args.out)
    print(f"Wrote drift report -> {args.out}")


if __name__ == "__main__":
    main()
