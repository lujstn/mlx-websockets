"""Tests for package installation and entry points."""

import subprocess
import sys
from importlib.metadata import entry_points
from unittest.mock import patch

import pytest


class TestPackageInstallation:
    """Test package installation and entry points."""

    def test_package_imports(self):
        """Test that all package modules can be imported."""
        import mlx_websockets
        import mlx_websockets.cli
        import mlx_websockets.daemon
        import mlx_websockets.server

        # Verify modules have expected attributes
        assert hasattr(mlx_websockets.cli, "main")
        assert hasattr(mlx_websockets.daemon, "start_background_server")
        assert hasattr(mlx_websockets.server, "main")

    def test_entry_point_exists(self):
        """Test that the mlx entry point is registered."""
        # Get all console_scripts entry points
        eps = entry_points()
        if hasattr(eps, "select"):
            # Python 3.10+
            scripts = eps.select(group="console_scripts")
        else:
            # Python 3.9
            scripts = eps.get("console_scripts", [])

        # Find our entry point
        mlx_entry = None
        for ep in scripts:
            if ep.name == "mlx":
                mlx_entry = ep
                break

        assert mlx_entry is not None, "mlx entry point not found"
        assert mlx_entry.value == "mlx_websockets.cli:main"

    def test_module_execution(self):
        """Test that the package can be run as a module."""
        # Test running with --help to avoid starting server
        result = subprocess.run(
            [sys.executable, "-m", "mlx_websockets", "--help"], capture_output=True, text=True
        )

        assert result.returncode == 0
        assert "usage: mlx" in result.stdout
        assert "Available commands" in result.stdout

    @pytest.mark.skipif(
        subprocess.run(["which", "mlx"], capture_output=True).returncode != 0,
        reason="mlx command not in PATH (package not installed)",
    )
    def test_command_line_entry(self):
        """Test that the mlx command works from command line."""
        result = subprocess.run(["mlx", "--help"], capture_output=True, text=True)

        assert result.returncode == 0
        assert "usage: mlx" in result.stdout

    def test_version_info(self):
        """Test that version information is available."""
        import mlx_websockets

        # Should have __version__ attribute
        assert hasattr(mlx_websockets, "__version__")
        # Version should be a string
        assert isinstance(mlx_websockets.__version__, str)
        # Basic version format check (e.g., "0.1.0")
        assert "." in mlx_websockets.__version__

    def test_dependencies_available(self):
        """Test that required dependencies are available."""
        required_packages = [
            "websockets",
            "mlx",
            "mlx_lm",
            "numpy",
            "pydantic",
        ]

        for package in required_packages:
            try:
                __import__(package.replace("-", "_"))
            except ImportError:
                pytest.fail(f"Required package '{package}' not available")

    def test_python_version_compatibility(self):
        """Test Python version requirements."""
        # Based on pyproject.toml requires-python = ">=3.9"
        assert sys.version_info >= (3, 9), "Python 3.9+ is required"

    def test_package_metadata(self):
        """Test package metadata is properly configured."""
        from importlib.metadata import metadata

        pkg_metadata = metadata("mlx-websockets")

        # Check required metadata fields
        assert pkg_metadata["Name"] == "mlx-websockets"
        assert "Summary" in pkg_metadata or "Description" in pkg_metadata
        assert "Author" in pkg_metadata or "Author-email" in pkg_metadata
        # License can be in different fields depending on PEP 639 adoption
        assert (
            "License" in pkg_metadata
            or "License-Expression" in pkg_metadata
            or pkg_metadata.get("License-File")
        )
        assert "Requires-Python" in pkg_metadata

        # Check Python version requirement
        assert pkg_metadata["Requires-Python"] == ">=3.9"

    def test_cli_help_subcommands(self):
        """Test that all CLI subcommands are accessible."""
        subcommands = ["serve", "background", "status", "help"]

        for cmd in subcommands:
            result = subprocess.run(
                [sys.executable, "-m", "mlx_websockets", cmd, "--help"],
                capture_output=True,
                text=True,
            )

            # help command returns 0, others might return 2 for --help
            assert result.returncode in [0, 2], f"Command '{cmd}' failed"
            assert "usage:" in result.stdout or "usage:" in result.stderr

    def test_background_subcommands(self):
        """Test background subcommands are accessible."""
        bg_subcommands = ["serve", "stop"]

        for subcmd in bg_subcommands:
            result = subprocess.run(
                [sys.executable, "-m", "mlx_websockets", "background", subcmd, "--help"],
                capture_output=True,
                text=True,
            )

            assert result.returncode in [0, 2], f"Background {subcmd} failed"
            assert "usage:" in result.stdout or "usage:" in result.stderr
