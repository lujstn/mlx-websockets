# Contributing to MLX WebSockets

Thank you for your interest in contributing to MLX WebSockets! This document provides guidelines and instructions for contributing.

## Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/lujstn/mlx-websockets.git
   cd mlx-websockets
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Run the setup script:
   ```bash
   ./scripts/setup.sh
   ```

   This will:
   - Install all dependencies
   - Set up pre-commit hooks
   - Run initial linting and formatting

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
- Linting checks (Black, Ruff)
- Tests across Python 3.9, 3.10, 3.11, and 3.12
- Conventional commit validation

Ensure all checks pass before requesting a review.