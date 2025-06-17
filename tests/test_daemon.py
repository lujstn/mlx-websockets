"""Tests for the daemon functionality."""

import json
import os
import signal
import socket
import subprocess
import tempfile
import time
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, Mock, mock_open, patch

import pytest

from mlx_websockets.daemon import (
    ProcessInfo,
    find_available_port,
    get_config_dir,
    get_config_file,
    get_log_file,
    get_pid_file,
    get_server_status,
    is_process_running,
    start_background_server,
    stop_background_server,
)
from mlx_websockets.exceptions import DaemonError


class TestDaemonPaths:
    """Test daemon path functions."""

    def test_get_config_dir(self):
        """Test config directory creation."""
        with patch("pathlib.Path.home") as mock_home:
            mock_home.return_value = Path("/home/test")
            with patch("pathlib.Path.mkdir") as mock_mkdir:
                config_dir = get_config_dir()

                assert config_dir == Path("/home/test/.mlx-websockets")
                mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)

    def test_get_pid_file(self):
        """Test PID file path."""
        with patch("mlx_websockets.daemon.get_config_dir") as mock_config_dir:
            mock_config_dir.return_value = Path("/test/config")

            pid_file = get_pid_file()
            assert pid_file == Path("/test/config/mlx-server.pid")

    def test_get_log_file(self):
        """Test log file path."""
        with patch("mlx_websockets.daemon.get_config_dir") as mock_config_dir:
            mock_config_dir.return_value = Path("/test/config")

            log_file = get_log_file()
            assert log_file == Path("/test/config/mlx-server.log")

    def test_get_config_file(self):
        """Test config file path."""
        with patch("mlx_websockets.daemon.get_config_dir") as mock_config_dir:
            mock_config_dir.return_value = Path("/test/config")

            config_file = get_config_file()
            assert config_file == Path("/test/config/mlx-server.json")


class TestPortFinding:
    """Test port finding functionality."""

    def test_find_available_port_first_available(self):
        """Test finding first available port."""
        mock_sock = MagicMock()
        with patch("socket.socket", return_value=mock_sock):
            port = find_available_port(8765)

            assert port == 8765
            mock_sock.bind.assert_called_once_with(("", 8765))
            mock_sock.close.assert_called_once()

    def test_find_available_port_skip_used(self):
        """Test skipping used ports."""
        mock_sock = MagicMock()
        # First call fails (port in use), second succeeds
        mock_sock.bind.side_effect = [OSError(), None]

        with patch("socket.socket", return_value=mock_sock):
            port = find_available_port(8765)

            assert port == 8766
            assert mock_sock.bind.call_count == 2
            assert mock_sock.close.call_count == 1  # Only closed on success

    def test_find_available_port_max_attempts(self):
        """Test failing to find port after max attempts."""
        mock_sock = MagicMock()
        mock_sock.bind.side_effect = OSError()

        with patch("socket.socket", return_value=mock_sock):
            with pytest.raises(RuntimeError, match="Could not find available port"):
                find_available_port(8765, max_attempts=5)

            assert mock_sock.bind.call_count == 5


class TestProcessUtils:
    """Test process utility functions."""

    def test_is_process_running_true(self):
        """Test when process is running."""
        with patch("os.kill") as mock_kill:
            assert is_process_running(12345) is True
            mock_kill.assert_called_once_with(12345, 0)

    def test_is_process_running_false(self):
        """Test when process is not running."""
        with patch("os.kill", side_effect=OSError):
            assert is_process_running(12345) is False


class TestServerStatus:
    """Test server status checking."""

    def test_get_server_status_no_files(self):
        """Test status when no files exist."""
        with patch("mlx_websockets.daemon.get_pid_file") as mock_pid_file:
            mock_pid_file.return_value.exists.return_value = False

            status = get_server_status()
            assert status is None

    def test_get_server_status_running(self):
        """Test status when server is running."""
        pid_content = "12345"
        config_content = json.dumps(
            {"port": 8765, "started": "2024-01-01T12:00:00", "model": "test-model"}
        )

        with (
            patch("mlx_websockets.daemon.get_pid_file") as mock_pid_file,
            patch("mlx_websockets.daemon.get_config_file") as mock_config_file,
            patch("mlx_websockets.daemon.is_process_running", return_value=True),
        ):
            mock_pid_file.return_value.exists.return_value = True
            mock_pid_file.return_value.read_text.return_value = pid_content
            mock_config_file.return_value.exists.return_value = True
            mock_config_file.return_value.read_text.return_value = config_content

            status = get_server_status()

            assert status == {
                "pid": 12345,
                "port": 8765,
                "started": "2024-01-01T12:00:00",
                "model": "test-model",
            }

    def test_get_server_status_stale_pid(self):
        """Test status cleanup when process is dead."""
        with (
            patch("mlx_websockets.daemon.get_pid_file") as mock_pid_file,
            patch("mlx_websockets.daemon.get_config_file") as mock_config_file,
            patch("mlx_websockets.daemon.is_process_running", return_value=False),
        ):
            mock_pid_file.return_value.exists.return_value = True
            mock_pid_file.return_value.read_text.return_value = "12345"
            mock_config_file.return_value.exists.return_value = True
            mock_config_file.return_value.read_text.return_value = "{}"

            status = get_server_status()

            assert status is None
            mock_pid_file.return_value.unlink.assert_called_once()
            mock_config_file.return_value.unlink.assert_called_once()

    def test_get_server_status_invalid_pid(self):
        """Test status with invalid PID file."""
        with (
            patch("mlx_websockets.daemon.get_pid_file") as mock_pid_file,
            patch("mlx_websockets.daemon.get_config_file") as mock_config_file,
        ):
            mock_pid_file.return_value.exists.return_value = True
            mock_pid_file.return_value.read_text.return_value = "invalid"
            mock_config_file.return_value.exists.return_value = True

            status = get_server_status()
            assert status is None


class TestStartBackgroundServer:
    """Test starting background server."""

    def test_start_server_already_running(self):
        """Test error when server is already running."""
        mock_process = ProcessInfo(pid=12345, port=8765, start_time=time.time(), is_daemon=True)

        with patch(
            "mlx_websockets.daemon.is_any_mlx_server_running", return_value=(True, mock_process)
        ):
            with patch(
                "mlx_websockets.daemon.get_server_status",
                return_value={"pid": 12345, "port": 8765, "model": "test-model"},
            ):
                with pytest.raises(DaemonError, match="already running"):
                    start_background_server({"model": "test-model", "port": 8765})

    @patch("mlx_websockets.daemon.subprocess.Popen")
    @patch("mlx_websockets.daemon.is_process_running")
    @patch("mlx_websockets.daemon.find_available_port")
    @patch("mlx_websockets.daemon.is_any_mlx_server_running")
    @patch("builtins.open", new_callable=mock_open)
    @patch("mlx_websockets.daemon.time.sleep")
    def test_start_server_success(
        self,
        mock_sleep,
        mock_file,
        mock_is_any_running,
        mock_find_port,
        mock_is_running,
        mock_popen,
    ):
        """Test successful server start."""
        mock_is_any_running.return_value = (False, None)
        mock_find_port.return_value = 8765
        mock_process = Mock()
        mock_process.pid = 12345
        mock_popen.return_value = mock_process
        mock_is_running.return_value = True

        with (
            patch("mlx_websockets.daemon.get_pid_file") as mock_pid_file,
            patch("mlx_websockets.daemon.get_config_file") as mock_config_file,
            patch("mlx_websockets.daemon.get_log_file") as mock_log_file,
            patch("mlx_websockets.daemon.get_registry") as mock_registry,
        ):
            mock_pid_file.return_value = Mock()
            mock_config_file.return_value = Mock()
            mock_log_file.return_value = Path("/test/mlx.log")
            mock_registry.return_value.register_process = Mock()

            start_background_server({"model": "test-model", "port": 8765})

            # start_background_server returns None, check via registry
            mock_registry.return_value.register_process.assert_called_once()
            mock_pid_file.return_value.write_text.assert_called_once_with("12345")

            # Verify command construction
            expected_cmd = [
                mock_popen.call_args[0][0][0],  # sys.executable
                "-m",
                "mlx_websockets.server",
                "--model",
                "test-model",
                "--port",
                "8765",
            ]
            assert mock_popen.call_args[0][0] == expected_cmd

    @patch("mlx_websockets.daemon.subprocess.Popen")
    @patch("mlx_websockets.daemon.is_process_running")
    @patch("mlx_websockets.daemon.is_any_mlx_server_running")
    @patch("builtins.open", new_callable=mock_open)
    @patch("mlx_websockets.daemon.time.sleep")
    def test_start_server_process_dies(
        self, mock_sleep, mock_file, mock_is_any_running, mock_is_running, mock_popen
    ):
        """Test error when server process dies immediately."""
        mock_is_any_running.return_value = (False, None)
        mock_process = Mock()
        mock_process.pid = 12345
        mock_popen.return_value = mock_process
        mock_is_running.return_value = False  # Process died

        with patch("mlx_websockets.daemon.get_log_file") as mock_log_file:
            mock_log_file.return_value = Path("/test/mlx.log")

            with pytest.raises(RuntimeError, match="Failed to start background server"):
                start_background_server({"model": "test-model", "port": 8765})

    def test_start_server_with_all_options(self):
        """Test starting server with all options."""
        with (
            patch("mlx_websockets.daemon.is_any_mlx_server_running", return_value=(False, None)),
            patch("mlx_websockets.daemon.subprocess.Popen") as mock_popen,
            patch("mlx_websockets.daemon.is_process_running", return_value=True),
            patch("builtins.open", new_callable=mock_open),
            patch("mlx_websockets.daemon.time.sleep"),
            patch("mlx_websockets.daemon.get_pid_file"),
            patch("mlx_websockets.daemon.get_config_file"),
            patch("mlx_websockets.daemon.get_log_file"),
            patch("mlx_websockets.daemon.get_registry"),
        ):
            mock_process = Mock()
            mock_process.pid = 12345
            mock_popen.return_value = mock_process

            start_background_server(
                {
                    "model": "model",
                    "host": "localhost",
                    "port": 9000,
                    "trust_remote_code": True,
                    "tokenizer_config": "config.json",
                    "chat_template": "template",
                    "max_tokens": 1024,
                    "temperature": 0.5,
                    "seed": 42,
                }
            )

            cmd = mock_popen.call_args[0][0]
            assert "--model" in cmd
            assert "--host" in cmd
            assert "--port" in cmd
            assert "--trust-remote-code" in cmd
            assert "--tokenizer-config" in cmd
            assert "--chat-template" in cmd
            assert "--max-tokens" in cmd
            assert "--temperature" in cmd
            assert "--seed" in cmd


class TestStopBackgroundServer:
    """Test stopping background server."""

    def test_stop_server_not_running(self):
        """Test stopping when no server is running."""
        with patch("mlx_websockets.daemon.get_server_status", return_value=None):
            with patch("mlx_websockets.daemon.find_all_mlx_processes", return_value=[]):
                assert stop_background_server() is False

    @patch("mlx_websockets.daemon.get_server_status")
    @patch("mlx_websockets.daemon.is_process_running")
    @patch("mlx_websockets.daemon.os.kill")
    @patch("mlx_websockets.daemon.time.sleep")
    @patch("mlx_websockets.daemon.get_pid_file")
    @patch("mlx_websockets.daemon.get_config_file")
    @patch("mlx_websockets.daemon.get_registry")
    @patch("mlx_websockets.daemon.find_all_mlx_processes")
    def test_stop_server_graceful(
        self,
        mock_find_all,
        mock_registry,
        mock_config_file,
        mock_pid_file,
        mock_sleep,
        mock_kill,
        mock_is_running,
        mock_status,
    ):
        """Test graceful server shutdown."""
        mock_status.return_value = {"pid": 12345}
        # Process stops after SIGTERM (first check in loop)
        mock_is_running.return_value = False
        mock_find_all.return_value = []

        # Mock file operations
        mock_pid_file.return_value = Mock()
        mock_config_file.return_value = Mock()
        mock_registry.return_value.unregister_process = Mock()

        assert stop_background_server() is True

        mock_kill.assert_called_once_with(12345, signal.SIGTERM)
        mock_pid_file.return_value.unlink.assert_called_once_with(missing_ok=True)
        mock_config_file.return_value.unlink.assert_called_once_with(missing_ok=True)

    @patch("mlx_websockets.daemon.get_server_status")
    @patch("mlx_websockets.daemon.is_process_running")
    @patch("mlx_websockets.daemon.os.kill")
    @patch("mlx_websockets.daemon.time.sleep")
    @patch("mlx_websockets.daemon.find_all_mlx_processes")
    def test_stop_server_force_kill(
        self, mock_find_all, mock_sleep, mock_kill, mock_is_running, mock_status
    ):
        """Test force killing server when graceful shutdown fails."""
        mock_status.return_value = {"pid": 12345}
        # Process keeps running until force kill
        mock_is_running.side_effect = [True] * 11 + [False]
        mock_find_all.return_value = []

        with (
            patch("mlx_websockets.daemon.get_pid_file") as mock_pid_file,
            patch("mlx_websockets.daemon.get_config_file") as mock_config_file,
            patch("mlx_websockets.daemon.get_registry"),
        ):
            mock_pid_file.return_value = Mock()
            mock_config_file.return_value = Mock()

            assert stop_background_server() is True

            # Should have called SIGTERM first, then SIGKILL
            assert mock_kill.call_count == 2
            mock_kill.assert_any_call(12345, signal.SIGTERM)
            mock_kill.assert_any_call(12345, signal.SIGKILL)

    @patch("mlx_websockets.daemon.get_server_status")
    @patch("mlx_websockets.daemon.os.kill")
    @patch("mlx_websockets.daemon.find_all_mlx_processes")
    def test_stop_server_exception(self, mock_find_all, mock_kill, mock_status):
        """Test handling exceptions during stop."""
        mock_status.return_value = {"pid": 12345}
        mock_kill.side_effect = OSError("Test error")
        mock_find_all.return_value = []

        # The function now continues after exceptions and returns True if status exists
        assert stop_background_server() is True
