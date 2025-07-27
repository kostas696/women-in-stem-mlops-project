# Makefile for Women in STEM MLOps Project

# === Variables ===
PYTHON := python3
PROJECT := women-in-stem-mlops-project

# === Setup ===
install:
	uv venv
	uv pip install -r requirements.txt
	uv pip install -e .

sync:
	uv pip sync

lint:
	ruff check .
	black .
	flake8 .

format:
	black .
	ruff format .

test:
	pytest tests/

preprocess:
	$(PYTHON) src/preprocess.py

train:
	$(PYTHON) src/train.py

evaluate:
	$(PYTHON) src/evaluate.py

inference:
	$(PYTHON) fastapi_app/main.py

docker-build:
	docker compose build

docker-up:
	docker compose up -d

docker-down:
	docker compose down

airflow-init:
	docker compose run airflow-init

mlflow-ui:
	mlflow ui --host 0.0.0.0 --port 5001

evidently-report:
	$(PYTHON) monitoring/generate_report.py

infra-up:
	cd infra && terraform init && terraform apply -auto-approve

infra-down:
	cd infra && terraform destroy -auto-approve

format-check:
	black --check .
	ruff check .

pre-commit:
	pre-commit run --all-files

# === All-in-One ===
run-all: preprocess train evaluate inference