from __future__ import annotations

import time
from typing import Any, Dict

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from prometheus_client import make_asgi_app

from .metrics import INFERENCE_LATENCY_SECONDS, REQUESTS_TOTAL, REQUEST_LATENCY_SECONDS
from .model_loader import load_model
from .utils import CIFAR10_CLASSES, append_inference_log, extract_drift_features, preprocess_image, softmax_probs

app = FastAPI(title="CV Inference API", version="1.0")

# Expose Prometheus metrics at /metrics
app.mount("/metrics", make_asgi_app())


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.middleware("http")
async def metrics_middleware(request, call_next):
    endpoint = request.url.path
    start = time.time()
    status = "200"
    try:
        response = await call_next(request)
        status = str(response.status_code)
        return response
    except Exception:
        status = "500"
        raise
    finally:
        dur = time.time() - start
        REQUEST_LATENCY_SECONDS.labels(endpoint=endpoint).observe(dur)
        REQUESTS_TOTAL.labels(endpoint=endpoint, status=status).inc()


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> Dict[str, Any]:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file")

    try:
        raw = await file.read()
        from PIL import Image
        import io

        img = Image.open(io.BytesIO(raw))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    x = preprocess_image(img)

    model = load_model()
    with INFERENCE_LATENCY_SECONDS.time():
        with torch.no_grad():
            logits = model(x)

    probs = softmax_probs(logits)
    pred_idx = int(probs.argmax())
    pred_label = CIFAR10_CLASSES[pred_idx]

    # Drift monitoring: record lightweight features
    feats = extract_drift_features(img)
    append_inference_log(feats, pred_idx)

    top5_idx = probs.argsort()[::-1][:5].tolist()
    top5 = [
        {"class": CIFAR10_CLASSES[i], "prob": float(probs[i])}
        for i in top5_idx
    ]

    return {
        "pred_class": pred_label,
        "pred_index": pred_idx,
        "top5": top5,
    }


@app.get("/model")
def model_info() -> Dict[str, str]:
    import os

    return {
        "mlflow_tracking_uri": os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"),
        "model_name": os.getenv("MODEL_NAME", "cifar10_cnn"),
        "model_stage": os.getenv("MODEL_STAGE", "Staging"),
    }
