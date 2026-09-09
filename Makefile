# Kokoro — developer entry points. `make check` is what CI runs.
.DEFAULT_GOAL := help
PY := .venv/bin/python

.PHONY: help venv install fmt lint types test cov check headers clean serve

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

venv: ## Create the virtualenv
	python3 -m venv .venv

install: venv ## Install the package with all extras
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -e ".[dev,index,serve]"
	$(PY) -m pre_commit install

fmt: ## Format the code
	$(PY) -m ruff format .
	$(PY) -m ruff check --fix .

lint: ## Lint without modifying anything
	$(PY) -m ruff format --check .
	$(PY) -m ruff check .

types: ## Type-check in strict mode
	$(PY) -m mypy

test: ## Run the test suite
	$(PY) -m pytest -q

cov: ## Run tests with an HTML coverage report
	$(PY) -m pytest --cov-report=html
	@echo "open htmlcov/index.html"

headers: ## Verify every source file carries its attribution header
	$(PY) scripts/check_headers.py

check: lint types headers test ## Everything CI runs

serve: ## Run the API locally
	$(PY) -m uvicorn kokoro.serve.app:app --reload --port 8000

clean: ## Remove build and cache artifacts
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
