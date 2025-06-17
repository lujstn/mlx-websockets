# Contributing to MLX WebSockets

Thank you for your interest in contributing to MLX WebSockets! This document provides guidelines and instructions for contributing.

## Development Setup

### Quick Setup (Recommended)

1. Clone the repository:
   ```bash
   git clone https://github.com/lujstn/mlx-websockets.git
   cd mlx-websockets
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Run the setup script:
   ```bash
   ./scripts/setup.sh
   ```

   This automated setup script will:
   - Install all development dependencies
   - Set up pre-commit hooks for code quality checks
   - Set up commit-msg hooks for conventional commit enforcement
   - Install a pre-push hook that runs all CI/CD checks locally
   - Run initial code quality checks on all files
   - Enable the `mlx` CLI command for testing

### Manual Setup (Alternative)

If you prefer to set up manually or the script doesn't work on your system:

1. Install the package in development mode:
   ```bash
   pip install -e ".[dev]"
   ```

2. Install pre-commit hooks:
   ```bash
   pre-commit install
   pre-commit install --hook-type commit-msg
   ```

3. Note: The manual setup won't install the pre-push hook that prevents CI/CD failures

## Running Tests

Run the test suite with:
```bash
pytest
```

For specific test categories:
```bash
# Unit tests only
pytest --ignore=tests/test_end_to_end.py

# With coverage report
pytest --cov=mlx_websockets --cov-report=html

# Run specific test file
pytest tests/test_cli.py -v
```

Before submitting PRs, ensure:
- All tests pass: `pytest`
- Code is formatted: `ruff format .`
- Linting passes: `ruff check .`
- Type checking passes: `mypy mlx_websockets --ignore-missing-imports`

The pre-push hook will automatically run all CI/CD checks before pushing:
- Code formatting check (`ruff format --check`)
- Linting (`ruff check`)
- Type checking (`mypy`)
- Unit tests (`pytest`)
- Build validation (`python -m build` and `twine check`)

To bypass the pre-push checks in emergencies: `git push --no-verify`

## Commit Guidelines

We use [Conventional Commits](https://www.conventionalcommits.org/) for all commit messages. The pre-commit hooks will enforce this automatically.

### Commit Types

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, semicolons, etc.)
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `test`: Adding or updating tests
- `build`: Build system changes
- `ci`: CI/CD changes
- `chore`: Other changes that don't modify src or test files

### Examples

```bash
git commit -m "feat: add support for streaming image models"
git commit -m "fix: handle websocket connection timeouts"
git commit -m "docs: update installation instructions"
```

## Code Style

We use:
- **Black** for code formatting (line length: 100)
- **Ruff** for linting
- **pytest** for testing

Run these commands before committing:
```bash
make format  # Auto-format code
make lint    # Check for linting issues
make test    # Run tests
```

Or run all at once:
```bash
make all
```

## Testing

- Write tests for all new features and bug fixes
- Ensure all tests pass before submitting a PR
- Aim for high test coverage

Run tests with coverage:
```bash
make test-cov
```

## Pull Request Process

1. Fork the repository
2. Create a feature branch: `git checkout -b feat/your-feature-name`
3. Make your changes and commit using conventional commits
4. Ensure all tests pass and code is properly formatted
5. Push to your fork and create a pull request
6. The PR title should follow conventional commits format

## CI/CD

All pull requests will automatically run:
- Linting checks (Ruff format & check, mypy)
- Tests across Python 3.9, 3.10, 3.11, and 3.12 on Ubuntu and macOS
- Build verification
- Installation tests
- Conventional commit validation

Ensure all checks pass before requesting a review.

## Release Process

Releases are automated through GitHub Actions when a new tag is pushed. Here's the process:

### 1. Update Version

Update the version in `mlx_websockets/__init__.py`:
```python
__version__ = "X.Y.Z"
```

Follow [semantic versioning](https://semver.org/):
- MAJOR version for incompatible API changes
- MINOR version for backwards-compatible functionality
- PATCH version for backwards-compatible bug fixes

### 2. Commit Version Change

```bash
git add mlx_websockets/__init__.py
git commit -m "chore: bump version to X.Y.Z"
```

### 3. Create and Push Tag

```bash
git tag -a vX.Y.Z -m "Release version X.Y.Z"
git push origin main --tags
```

### 4. Automated Release

The GitHub Actions workflow will automatically:
1. Run all tests on multiple Python versions and platforms
2. Build source distribution and wheel
3. Create a GitHub release with auto-generated release notes
4. Publish to PyPI (requires PyPI credentials in repository secrets)

### Manual Release (if needed)

If you need to release manually:
```bash
make release  # Shows release instructions
make build    # Build distribution packages
make publish  # Upload to PyPI (requires credentials)
```

### Pre-releases

For release candidates or beta versions:
```bash
# Version: X.Y.ZrcN or X.Y.ZbN
git tag -a vX.Y.ZrcN -m "Release candidate N for version X.Y.Z"
```

These will be marked as pre-releases on GitHub and can be published to Test PyPI first.

## Development Workflow Summary

1. **Setup**: Install with `pip install -e ".[dev]"` and `pre-commit install`
2. **Code**: Make your changes with proper tests
3. **Format**: Run `make format` to auto-format code
4. **Test**: Run `make test` to ensure tests pass
5. **Commit**: Use conventional commits
6. **Push**: Create PR and ensure CI passes
