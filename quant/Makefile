# MOEX Research Platform - Core Package

.PHONY: help install dev clean \
	ingest_moex build_curated data_quality build_universe \
	test lint type-check \
	run_dummy_backtest

help: ## Show this help message
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'

install: ## Install production dependencies
	pip install -e ".[dev]"

dev: ## Install all dependencies including optional
	pip install -e ".[dev,tracking,orchestration]"

clean: ## Clean build artifacts and cache
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/ dist/ .pytest_cache/ .mypy_cache/ .ruff_cache/
	rm -rf data/raw/moex_cache/

# =============================================================================
# Data Pipeline
# =============================================================================

ingest_moex: ## Download raw data from MOEX ISS
	python pipelines/ingest_moex.py --config configs/phase0_platform.yaml

build_curated: ## Build curated daily bars from raw data
	python pipelines/build_curated_data.py --config configs/phase0_platform.yaml

data_quality: ## Run data quality checks
	python -m src.data_quality.run_checks --config configs/phase0_platform.yaml

build_universe: ## Build investable universe
	python -m src.data_ingestion.build_universe --config configs/phase0_platform.yaml

# =============================================================================
# Testing & Quality
# =============================================================================

test: ## Run unit and integration tests
	pytest tests/ -v --tb=short --cov=src --cov-report=term-missing

test-unit: ## Run unit tests only
	pytest tests/unit/ -v --tb=short

test-integration: ## Run integration tests only
	pytest tests/integration/ -v --tb=short

lint: ## Run linter (ruff)
	ruff check src/ pipelines/ tests/

lint-fix: ## Fix linting issues automatically
	ruff check --fix src/ pipelines/ tests/

type-check: ## Run type checker (mypy)
	mypy src/ pipelines/

# =============================================================================
# Backtest & Reports
# =============================================================================

run_dummy_backtest: ## Run dummy backtest to verify platform
	python pipelines/run_backtest.py --config configs/phase0_platform.yaml --mode dummy

run_backtest: ## Run full backtest
	python pipelines/run_backtest.py --config configs/phase0_platform.yaml

generate_report: ## Generate metrics report
	python -m src.metrics.generate_report --config configs/phase0_platform.yaml

# =============================================================================
# Experiment Tracking (optional)
# =============================================================================

mlflow-server: ## Start MLflow tracking server
	mlflow ui --host 0.0.0.0 --port 5000
