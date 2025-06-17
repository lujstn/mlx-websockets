.PHONY: help
help:
	@echo "Available commands:"
	@echo "  make lint      - Run all linters (ruff format & check, mypy)"
	@echo "  make format    - Auto-format code with ruff"
	@echo "  make test      - Run all tests"
	@echo "  make test-cov  - Run tests with coverage"
	@echo "  make test-timing - Run tests with timing information"
	@echo "  make test-fast - Run only fast tests (exclude slow)"
	@echo "  make clean     - Remove Python cache files and build artifacts"
	@echo "  make build     - Build distribution packages"
	@echo "  make release   - Create a new release (tag, build, notes)"
	@echo "  make publish   - Publish to PyPI (requires authentication)"
	@echo "  make install   - Install package in development mode"
	@echo "  make all       - Run format, lint, and test"

.PHONY: lint
lint:
	@echo "Running ruff format check..."
	ruff format --check .
	@echo "Running ruff lint..."
	ruff check .
	@echo "Running mypy..."
	mypy mlx_websockets

.PHONY: format
format:
	@echo "Formatting code with ruff..."
	ruff format .
	@echo "Fixing imports with ruff..."
	ruff check --fix .

.PHONY: test
test:
	@echo "Running tests..."
	pytest -v

.PHONY: test-cov
test-cov:
	@echo "Running tests with coverage..."
	pytest -v --cov=mlx_websockets --cov-report=term-missing --cov-report=html

.PHONY: test-integration
test-integration:
	@echo "Running integration tests..."
	pytest -v tests/test_integration.py -m integration

.PHONY: test-performance
test-performance:
	@echo "Running performance benchmarks..."
	pytest -v -s tests/test_performance.py -m benchmark

.PHONY: test-fast
test-fast:
	@echo "Running fast tests only..."
	pytest -v -m "not slow and not benchmark"

.PHONY: test-timing
test-timing:
	@echo "Running tests with timing information..."
	pytest -v --durations=0

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
	@echo "Cleaning build artifacts..."
	rm -rf build/ dist/ .eggs/
	find . -name '*.egg' -exec rm -f {} +

.PHONY: install
install:
	@echo "Installing package in development mode..."
	pip install -e ".[dev]"

.PHONY: build
build: clean
	@echo "Building distribution packages..."
	python -m build
	@echo "Checking built packages..."
	twine check dist/*

.PHONY: release
release:
	@echo "Creating a new release..."
	@echo "Current version: $(shell python -c 'import mlx_websockets; print(mlx_websockets.__version__)')"
	@echo ""
	@echo "Steps to release:"
	@echo "1. Update version in mlx_websockets/_version.py"
	@echo "2. Commit changes: git commit -am 'chore: bump version to X.Y.Z'"
	@echo "3. Tag release: git tag -a vX.Y.Z -m 'Release version X.Y.Z'"
	@echo "4. Push changes: git push origin main --tags"
	@echo ""
	@echo "The GitHub Actions workflow will automatically:"
	@echo "- Build and test the package"
	@echo "- Create a GitHub release"
	@echo "- Update the Homebrew formula"

.PHONY: publish
publish:
	@echo "This project is distributed via Homebrew, not PyPI."
	@echo "To create a new release, use 'make release' and follow the instructions."

.PHONY: version
version:
	@python -c 'import mlx_websockets; print(mlx_websockets.__version__)'

.PHONY: all
all: format lint test
