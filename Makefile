.PHONY: help install dev lint format typecheck test test-unit test-integration \
        test-adversarial coverage docs docker security clean

PYTHON ?= python3

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

install:  ## Install the package
	$(PYTHON) -m pip install -e .

dev:  ## Install dev dependencies + pre-commit
	$(PYTHON) -m pip install -e ".[dev,monitoring]"

lint:  ## Ruff lint
	ruff check src/ tests/

format:  ## Auto-format with ruff
	ruff format src/ tests/
	ruff check src/ tests/ --fix

typecheck:  ## Mypy strict
	mypy src/

test:  ## Run the full test suite (excluding GPU-only) with coverage gate
	pytest -m "not gpu"

test-unit:  ## Unit tests only
	pytest tests/unit/ --no-cov

test-integration:  ## Integration tests
	pytest tests/integration/ --no-cov -m "not gpu"

test-adversarial:  ## Adversarial tests
	pytest tests/adversarial/ --no-cov

coverage:  ## Coverage report (>= 90% gate)
	pytest -m "not gpu" --cov=rlhf --cov-report=term-missing --cov-report=html

docs:  ## Build the docs site (strict)
	mkdocs build --strict -f docs/mkdocs.yml

docker:  ## Build the Docker image
	docker build -t rlhf-ppo:latest -f infra/Dockerfile .

security:  ## Run Bandit + pip-audit
	bandit -r src/ -ll
	pip-audit

clean:  ## Remove caches and build artifacts
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov coverage.xml dist build site
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
