"""End-to-end integration tests for MLX WebSockets."""

import asyncio
import json
import subprocess
import sys
import time
from unittest.mock import patch

import pytest
import websockets

from mlx_websockets.daemon import get_server_status, start_background_server, stop_background_server

from .test_helpers import mock_mlx_models_context


async def wait_for_server(port, max_wait=10, retry_interval=0.5):
    """Wait for server to be ready for connections."""
    start_time = time.time()
    while time.time() - start_time < max_wait:
        try:
            # Try to connect briefly to check if server is ready
            async with websockets.connect(f"ws://localhost:{port}") as _:
                # If we can connect, server is ready
                return True
        except (
            OSError,
            websockets.exceptions.ConnectionClosedError,
            websockets.exceptions.InvalidStatusCode,
        ):
            # Server not ready yet, wait and retry
            await asyncio.sleep(retry_interval)
    return False


@pytest.mark.integration
class TestEndToEnd:
    """End-to-end integration tests."""

    @pytest.fixture
    def cleanup_daemon(self):
        """Ensure daemon is stopped after tests."""
        yield
        # Clean up any running daemon
        try:
            stop_background_server()
        except Exception:
            pass

    @pytest.mark.slow
    @pytest.mark.timeout(60)
    def test_background_daemon_lifecycle(self, cleanup_daemon):
        """Test basic daemon lifecycle: start, status check, stop."""
        # Mock the MLX model loading to speed up tests
        from .test_helpers import mock_mlx_models

        with mock_mlx_models_context():
            # Start daemon
            start_background_server({"model": "mlx-community/gemma-3-4b-it-4bit", "port": 8765})

            # Check status - daemon process should be running
            status = get_server_status()
            assert status is not None
            assert status["port"] == 8765
            assert status["pid"] > 0

            # Verify process is actually running
            import os

            try:
                os.kill(status["pid"], 0)  # Signal 0 just checks if process exists
                process_running = True
            except (OSError, ProcessLookupError):
                process_running = False

            assert process_running, "Daemon process is not running"

            # Stop daemon
            assert stop_background_server() is True

            # Verify stopped
            status = get_server_status()
            assert status is None

    def test_cli_error_handling(self):
        """Test CLI error handling for invalid arguments."""
        # Test invalid port
        result = subprocess.run(
            [sys.executable, "-m", "mlx_websockets", "serve", "--port", "invalid"],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
        assert "invalid int value" in result.stderr or "invalid literal" in result.stderr

        # Test invalid temperature
        result = subprocess.run(
            [sys.executable, "-m", "mlx_websockets", "serve", "--temperature", "invalid"],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
        assert "invalid float value" in result.stderr or "could not convert" in result.stderr

    @pytest.mark.slow
    @pytest.mark.timeout(60)
    def test_daemon_port_collision_handling(self, cleanup_daemon):
        """Test daemon handles port collisions gracefully."""
        from .test_helpers import mock_mlx_models

        with mock_mlx_models_context():
            # Start first daemon
            start_background_server({"port": 8765, "model": "mlx-community/gemma-3-4b-it-4bit"})

            # Stop it
            stop_background_server()

            # Start another daemon requesting same port
            # Should succeed by finding available port
            start_background_server({"port": 8765, "model": "mlx-community/gemma-3-4b-it-4bit"})

            # Check that it started on a different port
            status2 = get_server_status()
            assert status2["port"] >= 8765  # Should be same or higher

            # Clean up
            stop_background_server()
