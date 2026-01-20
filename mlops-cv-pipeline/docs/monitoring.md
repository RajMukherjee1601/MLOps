# Monitoring

## Metrics exposed by the API

The FastAPI service exposes Prometheus metrics at:

- `GET /metrics`

Key metrics:
- `inference_requests_total{endpoint,status}`
- `inference_request_latency_seconds{endpoint}`
- `model_inference_latency_seconds`

## Local Prometheus (quick demo)

Create `prometheus.yml`:

```yaml
global:
  scrape_interval: 15s
scrape_configs:
  - job_name: cv-inference
    static_configs:
      - targets: ["host.docker.internal:8000"]
```

Run Prometheus:

```bash
docker run --rm -p 9090:9090 \
  -v $PWD/prometheus.yml:/etc/prometheus/prometheus.yml \
  prom/prometheus
```

Then visit http://localhost:9090 and query `inference_requests_total`.
