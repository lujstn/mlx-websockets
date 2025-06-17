#!/bin/bash
# Script to set up Homebrew tap for mlx-websockets

set -e

echo "Setting up Homebrew tap for mlx-websockets..."
echo ""

# Check if git is installed
if ! command -v git &> /dev/null; then
    echo "Error: git is not installed"
    exit 1
fi

# Check if GitHub CLI is installed (optional but helpful)
if command -v gh &> /dev/null; then
    echo "GitHub CLI detected. Creating repository..."

    # Create the tap repository on GitHub
    gh repo create homebrew-mlx-websockets --public --description "Homebrew tap for mlx-websockets" || true

    # Clone the repository
    git clone git@github.com:lujstn/homebrew-mlx-websockets.git
else
    echo "GitHub CLI not found. Please create the repository manually:"
    echo "  1. Go to https://github.com/new"
    echo "  2. Name it: homebrew-mlx-websockets"
    echo "  3. Make it public"
    echo "  4. Don't initialize with README"
    echo ""
    read -p "Press Enter once you've created the repository..."

    # Clone the repository
    git clone git@github.com:lujstn/homebrew-mlx-websockets.git
fi

cd homebrew-mlx-websockets

# Create the Formula directory
mkdir -p Formula

# Copy the formula
cp ../Formula/mlx-websockets.rb Formula/

# Create README
cat > README.md << 'EOF'
# Homebrew Tap for MLX WebSockets

This is the official Homebrew tap for [mlx-websockets](https://github.com/lujstn/mlx-websockets).

## Installation

```bash
brew tap lujstn/mlx-websockets
brew install mlx-websockets
```

## Usage

After installation, you can use the `mlx` command:

```bash
mlx serve                    # Run the WebSocket server
mlx background serve         # Run server in background
mlx background stop          # Stop background server
mlx status                   # Check server status
```

## Service Management

You can also run mlx-websockets as a service:

```bash
brew services start mlx-websockets
brew services stop mlx-websockets
brew services restart mlx-websockets
```

## Updating

To update to the latest version:

```bash
brew update
brew upgrade mlx-websockets
```

## Uninstallation

```bash
brew uninstall mlx-websockets
brew untap lujstn/mlx-websockets
```

## License

MIT License - see the main [mlx-websockets](https://github.com/lujstn/mlx-websockets) repository for details.
EOF

# Create .gitignore
cat > .gitignore << 'EOF'
.DS_Store
*.swp
*~
EOF

# Create GitHub Actions workflow for testing
mkdir -p .github/workflows
cat > .github/workflows/tests.yml << 'EOF'
name: Test Formula

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: macos-latest
    steps:
    - uses: actions/checkout@v4

    - name: Set up Homebrew
      run: |
        brew update

    - name: Test formula
      run: |
        brew audit --new --formula Formula/mlx-websockets.rb
        brew style Formula/mlx-websockets.rb
EOF

# Initialize git repository
git init
git add .
git commit -m "Initial commit: Add mlx-websockets formula"

# Push to GitHub
git branch -M main
git push -u origin main

echo ""
echo "✅ Homebrew tap setup complete!"
echo ""
echo "Your tap is now available at: https://github.com/lujstn/homebrew-mlx-websockets"
echo ""
echo "Users can now install mlx-websockets with:"
echo "  brew tap lujstn/mlx-websockets"
echo "  brew install mlx-websockets"
echo ""
echo "Next steps:"
echo "1. Test the installation locally:"
echo "   brew tap lujstn/mlx-websockets"
echo "   brew install mlx-websockets"
echo ""
echo "2. When you're ready for homebrew-core (after reaching 75+ stars):"
echo "   - Fork homebrew/homebrew-core"
echo "   - Add your formula"
echo "   - Submit a pull request"
