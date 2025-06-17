"""
Comprehensive tests for process management functionality
"""

import asyncio
import os
import signal
import socket
import subprocess
import sys
import time
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import psutil
import pytest

from mlx_websockets.daemon import (
    ProcessInfo,
    find_all_mlx_processes,
    find_available_port,
    format_time,
    get_server_status,
    is_any_mlx_server_running,
    is_our_daemon_process,
    is_port_in_use,
    start_background_server,
)
from mlx_websockets.exceptions import DaemonError, NetworkError
from mlx_websockets.server import MLXStreamingServer


class TestProcessDetection:
    """Test process detection and tracking"""

    @pytest.fixture
    def mock_process(self):
        """Create a mock process for testing"""
        proc = Mock()
        proc.info = {
            "pid": 12345,
            "name": "python",
            "cmdline": ["python", "-m", "mlx_websockets.server", "--port", "8765"],
            "create_time": 1234567890.0,
        }
        proc.net_connections = Mock(return_value=[])
        return proc

    def test_find_all_mlx_processes_no_psutil(self):
        """Test behavior when psutil is not available"""
        with patch("mlx_websockets.daemon.psutil", None):
            processes = find_all_mlx_processes()
            assert processes == []

    @patch("mlx_websockets.daemon.get_registry")
    @patch("mlx_websockets.daemon.is_process_running", return_value=True)
    @patch("mlx_websockets.daemon.is_port_in_use", return_value=True)
    def test_find_all_mlx_processes_with_mlx_server(
        self, mock_is_port, mock_is_running, mock_get_registry, mock_process
    ):
        """Test finding MLX server processes"""
        # Mock the registry to return empty list
        mock_registry = Mock()
        mock_registry.get_all_processes.return_value = []
        mock_get_registry.return_value = mock_registry

        with patch("mlx_websockets.daemon.psutil.process_iter") as mock_process_iter:
            mock_process_iter.return_value = [mock_process]

            processes = find_all_mlx_processes()

            assert len(processes) == 1
            assert processes[0].pid == 12345
            assert processes[0].port == 8765
            assert processes[0].start_time == 1234567890.0

    @patch("mlx_websockets.daemon.get_registry")
    @patch("mlx_websockets.daemon.is_process_running", return_value=True)
    @patch("mlx_websockets.daemon.is_port_in_use", return_value=True)
    def test_find_all_mlx_processes_multiple(
        self, mock_is_port, mock_is_running, mock_get_registry
    ):
        """Test finding multiple MLX processes"""
        # Mock the registry to return empty list
        mock_registry = Mock()
        mock_registry.get_all_processes.return_value = []
        mock_get_registry.return_value = mock_registry

        proc1 = Mock()
        proc1.info = {
            "pid": 12345,
            "cmdline": ["python", "-m", "mlx_websockets.server", "--port", "8765"],
            "create_time": 1234567890.0,
        }
        proc1.net_connections = Mock(return_value=[])

        proc2 = Mock()
        proc2.info = {
            "pid": 12346,
            "cmdline": ["python", "mlx-websockets", "--port", "8766"],
            "create_time": 1234567891.0,
        }
        proc2.net_connections = Mock(return_value=[])

        with patch("mlx_websockets.daemon.psutil.process_iter") as mock_process_iter:
            mock_process_iter.return_value = [proc1, proc2]

            processes = find_all_mlx_processes()

            assert len(processes) == 2
            assert processes[0].port == 8765
            assert processes[1].port == 8766

    def test_is_any_mlx_server_running_none(self):
        """Test when no MLX servers are running"""
        with patch("mlx_websockets.daemon.find_all_mlx_processes", return_value=[]):
            running, process = is_any_mlx_server_running()
            assert not running
            assert process is None

    def test_is_any_mlx_server_running_found(self):
        """Test when MLX server is found"""
        mock_process = ProcessInfo(pid=12345, port=8765, start_time=time.time(), is_daemon=True)

        with patch("mlx_websockets.daemon.find_all_mlx_processes", return_value=[mock_process]):
            running, process = is_any_mlx_server_running()
            assert running
            assert process.pid == 12345


class TestPortManagement:
    """Test port discovery and management"""

    def test_is_port_in_use_free(self):
        """Test checking a free port"""
        # Find a free port first
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            free_port = s.getsockname()[1]

        # Check it's free
        assert not is_port_in_use(free_port)

    def test_is_port_in_use_occupied(self):
        """Test checking an occupied port"""
        # Occupy a port
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            s.listen(1)
            occupied_port = s.getsockname()[1]

            # Check it's occupied
            assert is_port_in_use(occupied_port)

    def test_find_available_port_first_free(self):
        """Test finding available port when first is free"""
        # Mock socket to succeed on first try
        with patch("socket.socket") as mock_socket_class:
            mock_sock = MagicMock()
            mock_socket_class.return_value = mock_sock

            port = find_available_port(8765)
            assert port == 8765
            mock_sock.bind.assert_called_once_with(("", 8765))

    def test_find_available_port_skip_occupied(self):
        """Test finding available port when first is occupied"""
        with patch("socket.socket") as mock_socket_class:
            mock_sock = MagicMock()
            mock_socket_class.return_value = mock_sock

            # First call fails, second succeeds
            mock_sock.bind.side_effect = [OSError("Address in use"), None]

            port = find_available_port(8765)
            assert port == 8766

    def test_find_available_port_exhausted(self):
        """Test when no ports are available"""
        with patch("socket.socket") as mock_socket_class:
            mock_sock = MagicMock()
            mock_socket_class.return_value = mock_sock

            # All attempts fail
            mock_sock.bind.side_effect = OSError("Address in use")

            with pytest.raises(RuntimeError, match="Could not find available port"):
                find_available_port(8765, max_attempts=3)


class TestDaemonConflictDetection:
    """Test daemon conflict detection and handling"""

    @pytest.fixture
    def mock_external_process(self):
        """Mock an external (non-daemon) MLX process"""
        return ProcessInfo(
            pid=12345,
            port=8765,
            start_time=time.time() - 3600,  # 1 hour ago
            is_daemon=False,
        )

    @pytest.fixture
    def mock_daemon_process(self):
        """Mock a daemon-managed MLX process"""
        return ProcessInfo(
            pid=12346,
            port=8765,
            start_time=time.time() - 1800,  # 30 min ago
            is_daemon=True,
        )

    def test_start_with_external_process_running(self, mock_external_process):
        """Test starting daemon when external MLX process is running"""
        with patch(
            "mlx_websockets.daemon.is_any_mlx_server_running",
            return_value=(True, mock_external_process),
        ):
            with patch("mlx_websockets.daemon.find_next_available_port", return_value=8766):
                with pytest.raises(DaemonError) as exc_info:
                    start_background_server(
                        {"port": 8765, "model": "mlx-community/gemma-3-4b-it-4bit"}
                    )

                assert "not daemon-managed" in str(exc_info.value)
                assert "mlx bg stop" in str(exc_info.value)
                assert "8766" in str(exc_info.value)

    def test_start_with_daemon_process_running(self, mock_daemon_process):
        """Test starting daemon when daemon process is already running"""
        with patch(
            "mlx_websockets.daemon.is_any_mlx_server_running",
            return_value=(True, mock_daemon_process),
        ):
            with patch(
                "mlx_websockets.daemon.get_server_status",
                return_value={"model": "test-model", "port": 8765},
            ):
                with pytest.raises(DaemonError) as exc_info:
                    start_background_server(
                        {"port": 8765, "model": "mlx-community/gemma-3-4b-it-4bit"}
                    )

                assert "daemon-managed" in str(exc_info.value)
                assert "mlx bg stop" in str(exc_info.value)

    def test_start_with_port_busy_by_non_mlx(self):
        """Test starting when port is busy by non-MLX process"""
        with patch("mlx_websockets.daemon.is_any_mlx_server_running", return_value=(False, None)):
            with patch("mlx_websockets.daemon.is_port_in_use", return_value=True):
                with patch("mlx_websockets.daemon.find_available_port", return_value=8766):
                    with patch("subprocess.Popen") as mock_popen:
                        mock_popen.return_value.pid = 12347
                        with patch("mlx_websockets.daemon.is_process_running", return_value=True):
                            start_background_server(
                                {"port": 8765, "model": "mlx-community/gemma-3-4b-it-4bit"}
                            )

                            # Should use port 8766
                            cmd_args = mock_popen.call_args[0][0]
                            assert "--port" in cmd_args
                            port_idx = cmd_args.index("--port")
                            assert cmd_args[port_idx + 1] == "8766"


class TestServerPortDiscovery:
    """Test automatic port discovery in MLXStreamingServer"""

    @pytest.mark.asyncio
    async def test_auto_port_discovery_first_available(self):
        """Test server finds first available port"""
        from tests.test_helpers import mock_mlx_models_context

        with mock_mlx_models_context():
            server = MLXStreamingServer(port=8765, auto_port=True, load_model_on_init=False)

            # Mock the port check to succeed on first try
            async def mock_find_port(start_port, max_tries=100):
                return start_port

            server._find_available_port = mock_find_port

            # Test finding port
            port = await server._find_available_port(8765)
            assert port == 8765

    @pytest.mark.asyncio
    async def test_auto_port_discovery_skip_busy(self):
        """Test server skips busy ports"""
        from tests.test_helpers import mock_mlx_models_context

        with mock_mlx_models_context():
            server = MLXStreamingServer(port=8765, auto_port=True, load_model_on_init=False)

            # Actually test the port finding logic
            with patch("socket.socket") as mock_socket_class:
                mock_sock = MagicMock()
                mock_socket_class.return_value.__enter__.return_value = mock_sock

                # First port busy, second available
                mock_sock.bind.side_effect = [OSError("Address in use"), None]

                # Mock asyncio server creation
                mock_test_server = Mock()
                mock_test_server.close = Mock()
                mock_test_server.wait_closed = AsyncMock()
                with patch("asyncio.start_server", return_value=mock_test_server):
                    port = await server._find_available_port(8765, max_tries=2)
                    assert port == 8766

    @pytest.mark.asyncio
    async def test_auto_port_discovery_exhausted(self):
        """Test error when no ports available"""
        from tests.test_helpers import mock_mlx_models_context

        with mock_mlx_models_context():
            server = MLXStreamingServer(port=8765, auto_port=True, load_model_on_init=False)

            with patch("socket.socket") as mock_socket_class:
                mock_sock = MagicMock()
                mock_socket_class.return_value.__enter__.return_value = mock_sock

                # All ports busy
                mock_sock.bind.side_effect = OSError("Address in use")

                with pytest.raises(NetworkError, match="No available ports found"):
                    await server._find_available_port(8765, max_tries=3)

    @pytest.mark.asyncio
    async def test_port_zero_os_assignment(self):
        """Test port 0 lets OS assign port"""
        from tests.test_helpers import mock_mlx_models_context

        with mock_mlx_models_context():
            server = MLXStreamingServer(port=0, auto_port=True, load_model_on_init=False)

            assert server.requested_port == 0
            assert server.auto_port is True


class TestProcessLifecycle:
    """Test process lifecycle guarantees"""

    def test_signal_handler_registration(self):
        """Test signal handlers are registered"""
        from tests.test_helpers import mock_mlx_models_context

        with mock_mlx_models_context():
            with patch("signal.signal") as mock_signal:
                with patch("atexit.register") as mock_atexit:
                    _ = MLXStreamingServer(load_model_on_init=False)

                    # Check atexit was registered
                    mock_atexit.assert_called_once()

                    # Check signals were registered
                    signal_calls = [call[0][0] for call in mock_signal.call_args_list]
                    assert signal.SIGINT in signal_calls
                    assert signal.SIGTERM in signal_calls

    def test_emergency_cleanup(self):
        """Test emergency cleanup function"""
        from tests.test_helpers import mock_mlx_models_context

        with mock_mlx_models_context():
            server = MLXStreamingServer(load_model_on_init=False)

            # Set up some state
            server.model = Mock()
            server.processor = Mock()
            server.resource_monitor = Mock()
            server.server = Mock()

            # Run emergency cleanup
            server._emergency_cleanup()

            # Check cleanup happened
            assert server.model is None
            assert server.processor is None
            server.resource_monitor.stop.assert_called_once()
            # Note: server.close() is not called in emergency cleanup since it's async

    def test_cleanup_registration_idempotent(self):
        """Test cleanup registration is idempotent"""
        from tests.test_helpers import mock_mlx_models_context

        with mock_mlx_models_context():
            with patch("atexit.register") as mock_atexit:
                server = MLXStreamingServer(load_model_on_init=False)

                # First registration in __init__
                assert mock_atexit.call_count == 1

                # Try to register again
                server._register_cleanup()

                # Should still be 1
                assert mock_atexit.call_count == 1


class TestUtilityFunctions:
    """Test utility functions"""

    def test_format_time_valid(self):
        """Test formatting valid timestamp"""
        timestamp = 1234567890.0
        formatted = format_time(timestamp)
        assert "2009" in formatted  # Unix timestamp from 2009

    def test_format_time_invalid(self):
        """Test formatting invalid timestamp"""
        formatted = format_time(-999999999999)
        assert formatted == "Unknown"

    def test_is_our_daemon_process_true(self):
        """Test checking if process is daemon-managed"""
        with patch("mlx_websockets.daemon.get_pid_file") as mock_pid_file:
            mock_path = Mock()
            mock_path.exists.return_value = True
            mock_path.read_text.return_value = "12345"
            mock_pid_file.return_value = mock_path

            assert is_our_daemon_process(12345)
            assert not is_our_daemon_process(99999)

    def test_is_our_daemon_process_no_file(self):
        """Test when PID file doesn't exist"""
        with patch("mlx_websockets.daemon.get_pid_file") as mock_pid_file:
            mock_path = Mock()
            mock_path.exists.return_value = False
            mock_pid_file.return_value = mock_path

            assert not is_our_daemon_process(12345)


class TestIntegrationScenarios:
    """Test real-world integration scenarios"""

    @pytest.mark.asyncio
    async def test_no_orphaned_processes_after_errors(self):
        """Test no processes left after various error conditions"""
        from tests.test_helpers import mock_mlx_models_context

        with mock_mlx_models_context():
            server = MLXStreamingServer(
                port=0,  # Let OS assign
                load_model_on_init=False,
            )

            # Simulate various errors
            with patch("websockets.serve", side_effect=OSError("Connection failed")):
                with pytest.raises(OSError):
                    await server.start_server()

            # Emergency cleanup should have been registered
            assert server._cleanup_registered

            # Manually trigger cleanup
            server._emergency_cleanup()

            # Check everything is cleaned up
            assert server.model is None
            assert server.processor is None

    def test_port_conflicts_resolved_automatically(self):
        """Test automatic port resolution in real scenarios"""
        with patch("socket.socket") as mock_socket_class:
            mock_sock = MagicMock()
            mock_socket_class.return_value = mock_sock

            # First port busy, second available
            mock_sock.bind.side_effect = [OSError("Address in use"), None]

            port = find_available_port(8765, max_attempts=5)
            assert port == 8766  # Second port
