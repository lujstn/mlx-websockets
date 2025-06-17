#!/bin/bash
# Script to fetch and calculate SHA256 hashes for Homebrew formula resources

set -e

echo "Fetching SHA256 hashes for Homebrew formula resources..."
echo ""

# Define resources with their PyPI URLs
RESOURCES=(
  "mlx|https://files.pythonhosted.org/packages/source/m/mlx/mlx-0.26.0.tar.gz"
  "mlx-lm|https://files.pythonhosted.org/packages/source/m/mlx-lm/mlx-lm-0.25.0.tar.gz"
  "mlx-vlm|https://files.pythonhosted.org/packages/source/m/mlx-vlm/mlx-vlm-0.1.26.tar.gz"
  "websockets|https://files.pythonhosted.org/packages/source/w/websockets/websockets-15.0.1.tar.gz"
  "Pillow|https://files.pythonhosted.org/packages/source/P/Pillow/Pillow-11.2.1.tar.gz"
  "numpy|https://files.pythonhosted.org/packages/source/n/numpy/numpy-1.26.4.tar.gz"
)

# Output that can be copied into the formula
echo "# Resource definitions for homebrew-formula.rb"
echo "# Generated on $(date)"
echo ""

for resource in "${RESOURCES[@]}"; do
  IFS='|' read -r name url <<< "$resource"
  echo "Fetching $name from $url..."
  sha256=$(curl -sL "$url" | shasum -a 256 | awk '{print $1}')

  echo ""
  echo "  resource \"$name\" do"
  echo "    url \"$url\""
  echo "    sha256 \"$sha256\""
  echo "  end"
  echo ""
done

echo "# End of resource definitions"
