# CV MLOps Pipeline (Train → Registry → Deploy → Monitor)

A portfolio-ready, end-to-end **MLOps** project for a computer-vision classifier:

- **Train** a small CV model (CIFAR-10) with PyTorch
- **Track** experiments with **MLflow**
- **Register** the model in **MLflow Model Registry**
- **Serve** the production model via **FastAPI** (Docker-ready)
- **CI/CD** with GitHub Actions (tests + Docker build + optional K8s deploy)
- **Monitor** latency & request volume via **Prometheus metrics**
- **Detect drift** with an offline drift report job (Evidently)

> This repo is designed to run locally with Docker first, and then on Kubernetes.

---

## 0) Prerequisites

- Python 3.10+ (3.11 works)
- Docker + Docker Compose
- (Optional) kubectl + a Kubernetes cluster (minikube/kind/EKS/GKE)

---

## 1) Quick start (Local)

### A. Start MLflow (tracking + registry)

```bash
cd mlflow
docker compose up -d
```

MLflow UI: http://localhost:5000

### B. Create a venv & install deps

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### C. Train + register a model

```bash
export MLFLOW_TRACKING_URI=http://localhost:5000
export MLFLOW_EXPERIMENT_NAME=cv-cifar10
export MODEL_NAME=cifar10_cnn

python -m src.training.train \
  --epochs 1 \
  --batch-size 64 \
  --register
```

This will:
- log metrics/artifacts to MLflow
- register the model under `MODEL_NAME`
- (by default) move the newest version to **Staging**

### D. Run the inference API (loads from MLflow Registry)

```bash
export MLFLOW_TRACKING_URI=http://localhost:5000
export MODEL_NAME=cifar10_cnn
export MODEL_STAGE=Staging

uvicorn src.serving.app:app --host 0.0.0.0 --port 8000
```

Try it:

```bash
curl -s http://localhost:8000/health
```

Send an image:

```bash
curl -s -X POST \
  -F "file=@scripts/sample_cifar10.png" \
  http://localhost:8000/predict
```

Prometheus metrics:

```bash
curl -s http://localhost:8000/metrics | head
```

---

## 2) Docker

### Build and run the API container

```bash
docker build -f docker/Dockerfile.serve -t cv-inference:local .

docker run --rm -p 8000:8000 \
  -e MLFLOW_TRACKING_URI=http://host.docker.internal:5000 \
  -e MODEL_NAME=cifar10_cnn \
  -e MODEL_STAGE=Staging \
  cv-inference:local
```

---

## 3) CI/CD (GitHub Actions)

Workflows are in `.github/workflows/`:

- **ci.yml**: lint + unit tests
- **cd.yml**: build & push Docker image to GHCR (optional deploy step)

To enable pushing images:
- Create `GHCR_TOKEN` with `write:packages`
- Add as a repo secret

---

## 4) Kubernetes (optional)

Manifests in `k8s/`:

```bash
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/
```

Set these environment variables inside `k8s/deployment.yaml`:
- `MLFLOW_TRACKING_URI` (your MLflow server)
- `MODEL_NAME`
- `MODEL_STAGE`

---

## 5) Drift monitoring (offline job)

The inference service appends minimal request features to `./data/inference_logs.jsonl` (volume-mount in prod).

Generate a drift report:

```bash
python -m src.monitoring.drift_report \
  --baseline data/baseline_features.jsonl \
  --current data/inference_logs.jsonl \
  --out reports/drift_report.html
```

---

## 6) What to highlight on your resume

- Built end-to-end CV MLOps pipeline: training → MLflow registry → containerized FastAPI inference
- Implemented CI/CD (tests + docker build + push to registry) with GitHub Actions
- Added production monitoring: Prometheus metrics for latency, throughput, errors
- Implemented drift reporting for incoming inference data

---

## License

MIT
