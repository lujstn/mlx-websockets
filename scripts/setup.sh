#!/bin/bash

echo "Setting up MLX WebSocket development environment..."

# Install development dependencies
echo "Installing development dependencies..."
pip install -e ".[dev]"

# Install pre-commit hooks
echo "Installing pre-commit hooks..."
pre-commit install
pre-commit install --hook-type commit-msg

# Install pre-push hook
echo "Installing pre-push hook..."
PRE_PUSH_HOOK=".git/hooks/pre-push"
if [ -f "$PRE_PUSH_HOOK" ]; then
    echo "Pre-push hook already exists, backing up..."
    mv "$PRE_PUSH_HOOK" "$PRE_PUSH_HOOK.backup"
fi

# Create pre-push hook
cat > "$PRE_PUSH_HOOK" << 'EOF'
#!/bin/bash
# Pre-push hook for mlx-websockets
# Runs all CI/CD checks before pushing to ensure code quality

set -e  # Exit on first error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Progress indicator
echo -e "${BLUE}Running pre-push checks...${NC}"
echo ""

# Function to print step
print_step() {
    echo -e "${YELLOW}▶ $1${NC}"
}

# Function to print success
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

# Function to print error
print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# Get the root directory of the git repository
ROOT_DIR=$(git rev-parse --show-toplevel)
cd "$ROOT_DIR"

# Create a temporary directory for build artifacts
TEMP_DIR=$(mktemp -d)
trap "rm -rf $TEMP_DIR" EXIT

# 1. Check formatting with ruff
print_step "Checking code formatting with ruff..."
if ruff format --check . > /dev/null 2>&1; then
    print_success "Code formatting check passed"
else
    print_error "Code formatting check failed"
    echo "Run 'ruff format .' to fix formatting issues"
    exit 1
fi

# 2. Run ruff linting
print_step "Running ruff linting..."
if ruff check .; then
    print_success "Linting passed"
else
    print_error "Linting failed"
    exit 1
fi

# 3. Run mypy type checking
print_step "Running mypy type checking..."
if mypy mlx_websockets --ignore-missing-imports; then
    print_success "Type checking passed"
else
    print_error "Type checking failed"
    exit 1
fi

# 4. Run unit tests
print_step "Running unit tests..."
if pytest tests/ -v --tb=short --ignore=tests/test_end_to_end.py; then
    print_success "Unit tests passed"
else
    print_error "Unit tests failed"
    exit 1
fi

# 5. Check build
print_step "Checking build..."
# Clean previous builds
rm -rf dist/ build/ *.egg-info

# Build the package
if python -m build --outdir "$TEMP_DIR" > /dev/null 2>&1; then
    print_success "Build succeeded"

    # Check with twine
    print_step "Checking distribution with twine..."
    if twine check "$TEMP_DIR"/*; then
        print_success "Distribution check passed"
    else
        print_error "Distribution check failed"
        exit 1
    fi
else
    print_error "Build failed"
    exit 1
fi

# 6. Quick import test
print_step "Testing imports..."
if python -c "import mlx_websockets; print(f'Version: {mlx_websockets.__version__}')" > /dev/null 2>&1; then
    print_success "Import test passed"
else
    print_error "Import test failed"
    exit 1
fi

echo ""
echo -e "${GREEN}All pre-push checks passed! ✅${NC}"
echo -e "${BLUE}Proceeding with push...${NC}"
echo ""

# Allow the push to proceed
exit 0
EOF

chmod +x "$PRE_PUSH_HOOK"
echo "Pre-push hook installed successfully!"

# Run pre-commit on all files
echo "Running pre-commit on all files..."
pre-commit run --all-files

echo "Setup complete! You're ready to start developing."
echo ""
echo "Important commands:"
echo "  mlx serve     - Run the WebSocket server"
echo "  mlx help      - Show CLI help"
echo "  make lint     - Run linters"
echo "  make format   - Auto-format code"
echo "  make test     - Run tests"
echo "  make all      - Run format, lint, and test"
echo ""
echo "Pre-commit hooks are now installed and will run automatically on commits."
echo "Pre-push hook is installed and will run all CI/CD checks before pushing."
echo ""
echo "To bypass pre-push checks in emergencies, use: git push --no-verify"
