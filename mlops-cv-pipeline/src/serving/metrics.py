from __future__ import annotations

from prometheus_client import Counter, Histogram

REQUESTS_TOTAL = Counter(
    "inference_requests_total",
    "Total number of inference requests",
    ["endpoint", "status"],
)

REQUEST_LATENCY_SECONDS = Histogram(
    "inference_request_latency_seconds",
    "Request latency in seconds",
    ["endpoint"],
)

INFERENCE_LATENCY_SECONDS = Histogram(
    "model_inference_latency_seconds",
    "Model inference latency in seconds",
)
