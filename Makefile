.PHONY: help
help:
	@echo "Available commands:"
	@echo "  make lint      - Run all linters (black, ruff)"
	@echo "  make format    - Auto-format code with black"
	@echo "  make test      - Run all tests"
	@echo "  make test-cov  - Run tests with coverage"
	@echo "  make clean     - Remove Python cache files"
	@echo "  make all       - Run format, lint, and test"

.PHONY: lint
lint:
	@echo "Running black (check mode)..."
	black --check .
	@echo "Running ruff..."
	ruff check .

.PHONY: format
format:
	@echo "Formatting code with black..."
	black .
	@echo "Fixing imports with ruff..."
	ruff check --fix .

.PHONY: test
test:
	@echo "Running tests..."
	pytest -v

.PHONY: test-cov
test-cov:
	@echo "Running tests with coverage..."
	pytest -v --cov=. --cov-report=term-missing --cov-report=html

.PHONY: clean
clean:
	@echo "Cleaning Python cache files..."
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name "htmlcov" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +

.PHONY: all
all: format lint test