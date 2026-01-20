.PHONY: help venv install mlflow-up mlflow-down train serve test lint docker-build docker-run

help:
	@echo "Targets: venv install mlflow-up mlflow-down train serve test lint docker-build docker-run"

venv:
	python -m venv .venv
	@echo "Run: source .venv/bin/activate"

install:
	pip install -r requirements.txt

mlflow-up:
	cd mlflow && docker compose up -d

mlflow-down:
	cd mlflow && docker compose down -v

train:
	python -m src.training.train --epochs 1 --batch-size 64 --register

serve:
	uvicorn src.serving.app:app --host 0.0.0.0 --port 8000

test:
	pytest -q

lint:
	ruff check .

docker-build:
	docker build -f docker/Dockerfile.serve -t cv-inference:local .

docker-run:
	docker run --rm -p 8000:8000 \
		-e MLFLOW_TRACKING_URI=http://host.docker.internal:5000 \
		-e MODEL_NAME=cifar10_cnn \
		-e MODEL_STAGE=Staging \
		cv-inference:local
