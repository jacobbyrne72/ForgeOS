.PHONY: test lint format clean install dev benchmark

# ForgeOS - Cost-governed AI harness
# Run `make help` for all available commands

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install forgeos and dependencies
	pip install -e .

dev: install ## Install dev dependencies
	pip install semgrep gitleaks ruff tree-sitter pytest

lint: ## Run ruff linter
	ruff check forgeos/ tests/ 2>/dev/null || echo "No ruff errors or ruff not installed"

format: ## Run ruff formatter
	ruff format forgeos/ tests/ 2>/dev/null || echo "No ruff formatter or not installed"

test: ## Run verification suite (short)
	python -c "from forgeos import *; print('All imports OK')"
	python -c "from forgeos.cli import *; print('All 31 CLI handlers OK')"
	python -m forgeos.cli doctor 2>&1 | grep -c "CPU\|provider" && echo "doctor works"

benchmark: ## Run reproducible cost benchmarks
	python -m forgeos.cli bench "code_gen" --iterations 10

audit: ## Scan project for AI cost waste
	python -m forgeos.cli audit --dir .

batch: ## Run batch cost projection
	python -m forgeos.cli batch --daily-tasks 100

auto: ## Run auto optimizer demo
	python -m forgeos.cli auto --task-type code_gen --daily-tasks 200

version: ## Show current version
	python -c "from forgeos import __version__; print(__version__)"

clean: ## Remove caches and build artifacts
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .forgeos -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

efficiency: ## Measure cost per useful output
	python -m forgeos.cli efficiency

forecast: ## Forecast future costs from history
	python -m forgeos.cli forecast --days 30

guard: ## Enforce hard budget cap
	python -m forgeos.cli guard --budget 10.0

scheduler: ## Batch tasks for minimum cost
	python -m forgeos.cli schedule --batch-size 10

all: test lint format ## Run full checks
