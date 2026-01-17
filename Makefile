.PHONY: help setup install dev-install docker-up docker-down extract-data process-data train-baseline evaluate api test notebook lint format clean

PYTHON := python
UV := uv

help:
	@echo "Available commands:"
	@echo "  make setup          - Install dependencies and start Docker services"
	@echo "  make install        - Install project dependencies with uv"
	@echo "  make dev-install    - Install with dev dependencies"
	@echo "  make docker-up      - Start Docker services (PostgreSQL, Redis, ClickHouse)"
	@echo "  make docker-down    - Stop Docker services"
	@echo "  make extract-data   - Extract RetailRocket dataset"
	@echo "  make process-data   - Run data processing pipeline"
	@echo "  make train-baseline - Train baseline models"
	@echo "  make evaluate       - Evaluate all models and print metrics"
	@echo "  make api            - Run FastAPI server"
	@echo "  make test           - Run tests"
	@echo "  make notebook       - Start Jupyter Lab"
	@echo "  make lint           - Run linter"
	@echo "  make format         - Format code"
	@echo "  make clean          - Clean generated files"

# Setup
setup: install docker-up
	@echo "Setup complete!"

install:
	$(UV) sync

dev-install:
	$(UV) sync --all-extras

# Docker
docker-up:
	docker compose up -d
	@echo "Waiting for services to be healthy..."
	@sleep 5

docker-down:
	docker compose down

# Data Pipeline
extract-data:
	$(UV) run $(PYTHON) scripts/extract_data.py

process-data:
	$(UV) run $(PYTHON) scripts/process_data.py

# Training
train-baseline:
	$(UV) run $(PYTHON) scripts/train_baseline.py

# Evaluation
evaluate:
	$(UV) run $(PYTHON) scripts/evaluate_models.py

# API
api:
	$(UV) run uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# Testing
test:
	$(UV) run pytest tests/ -v

test-cov:
	$(UV) run pytest tests/ -v --cov=src --cov-report=term-missing

# Notebook
notebook:
	$(UV) run jupyter lab

# Code Quality
lint:
	$(UV) run ruff check src/ tests/

format:
	$(UV) run ruff format src/ tests/
	$(UV) run ruff check --fix src/ tests/

# Cleanup
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .coverage htmlcov/ 2>/dev/null || true
