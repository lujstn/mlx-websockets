#!/bin/bash

echo "Setting up MLX WebSocket development environment..."

# Install development dependencies
echo "Installing development dependencies..."
pip install -e ".[dev]"

# Install pre-commit hooks
echo "Installing pre-commit hooks..."
pre-commit install
pre-commit install --hook-type commit-msg

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
