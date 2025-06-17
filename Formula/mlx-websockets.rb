class MlxWebsockets < Formula
  include Language::Python::Virtualenv

  desc "WebSocket streaming server for MLX models on Apple Silicon"
  homepage "https://github.com/lujstn/mlx-websockets"
  url "https://github.com/lujstn/mlx-websockets/archive/refs/tags/v0.1.0.tar.gz"
  sha256 "PLACEHOLDER_SHA256"  # This will be updated by the release workflow
  license "MIT"

  depends_on "python@3.11"
  depends_on "rust" => :build  # Required for some Python dependencies

  # MLX requires macOS and Apple Silicon
  depends_on :macos
  depends_on arch: :arm64

  # Python dependencies will be automatically resolved by pip
  # when installing from PyPI package

  def install
    # Install Python package with all dependencies
    virtualenv_install_with_resources

    # Ensure the mlx command is available
    bin.install_symlink libexec/"bin/mlx"
  end

  service do
    run [opt_bin/"mlx", "serve"]
    keep_alive true
    log_path var/"log/mlx-websockets.log"
    error_log_path var/"log/mlx-websockets.log"
  end

  test do
    # Test the CLI is accessible
    system bin/"mlx", "--help"

    # Test status command
    system bin/"mlx", "status"

    # Test Python module import
    system libexec/"bin/python", "-c", "import mlx_websockets; print(mlx_websockets.__version__)"
  end

  def caveats
    <<~EOS
      MLX WebSockets has been installed!

      Quick start:
        mlx serve                    # Run the WebSocket server
        mlx background serve         # Run server in background
        mlx background stop          # Stop background server
        mlx status                   # Check server status

      The server runs on port 8765 by default. To use a different port:
        mlx serve --port 8080

      To run as a background service with Homebrew:
        brew services start mlx-websockets
    EOS
  end
end
