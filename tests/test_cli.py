"""Tests for the CLI interface."""

import sys
from io import StringIO
from unittest.mock import MagicMock, patch

import pytest

from mlx_websockets.cli import main
from mlx_websockets.daemon import ProcessInfo


class TestCLI:
    """Test the command-line interface."""

    def test_no_command_shows_help(self, capsys):
        """Test that running without a command shows help."""
        with pytest.raises(SystemExit) as exc_info:
            main([])
        assert exc_info.value.code == 0

        captured = capsys.readouterr()
        assert "usage: mlx" in captured.out
        assert "Available commands" in captured.out

    def test_help_command(self, capsys):
        """Test the help command."""
        with patch("mlx_websockets.cli.sys.exit"):
            main(["help"])

        captured = capsys.readouterr()
        assert "usage: mlx" in captured.out
        assert "Available commands" in captured.out

    def test_invalid_command(self, capsys):
        """Test handling of invalid command."""
        with pytest.raises(SystemExit) as exc_info:
            main(["invalid"])
        assert exc_info.value.code == 2

        captured = capsys.readouterr()
        assert "invalid choice: 'invalid'" in captured.err

    @patch("mlx_websockets.daemon.find_all_mlx_processes")
    @patch("mlx_websockets.cli.run_server")
    def test_serve_command_default_args(self, mock_run_server, mock_find_all):
        """Test serve command with default arguments."""
        # Mock no processes running
        mock_find_all.return_value = []

        # Store original sys.argv
        original_argv = sys.argv
        try:
            main(["serve"])

            mock_run_server.assert_called_once()
            # The serve_command function should have modified sys.argv
            # before calling run_server
        finally:
            # Restore original sys.argv
            sys.argv = original_argv

    @patch("mlx_websockets.daemon.find_all_mlx_processes")
    @patch("mlx_websockets.cli.run_server")
    @patch("mlx_websockets.cli.sys.argv", ["mlx"])
    def test_serve_command_with_all_args(self, mock_run_server, mock_find_all):
        """Test serve command with all arguments."""
        args = [
            "serve",
            "--model",
            "test-model",
            "--host",
            "localhost",
            "--port",
            "9999",
            "--trust-remote-code",
            "--tokenizer-config",
            "config.json",
            "--chat-template",
            "template",
            "--max-tokens",
            "1024",
            "--temperature",
            "0.5",
            "--seed",
            "42",
        ]

        # Mock no processes running
        mock_find_all.return_value = []

        main(args)

        mock_run_server.assert_called_once()
        # The mock should have been called after sys.argv was modified

    @patch("mlx_websockets.cli.start_background_server")
    @patch("mlx_websockets.cli.get_server_status")
    def test_background_serve_command(self, mock_status, mock_start):
        """Test background serve command."""
        mock_start.return_value = None  # start_background_server returns None
        mock_status.return_value = {"port": 8765}

        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            main(["background", "serve", "--model", "test-model"])

        mock_start.assert_called_once()
        call_args = mock_start.call_args[0][0]  # Get the dictionary argument
        assert call_args["model"] == "test-model"

        output = mock_stdout.getvalue()
        assert "Server started in background on port 8765" in output
        assert "To stop: mlx background stop" in output

    @patch("mlx_websockets.cli.start_background_server")
    @patch("mlx_websockets.cli.get_server_status")
    def test_bg_alias_serve_command(self, mock_status, mock_start):
        """Test that 'bg' alias works for background command."""
        mock_start.return_value = None  # start_background_server returns None
        mock_status.return_value = {"port": 8765}

        main(["bg", "serve"])
        mock_start.assert_called_once()

    @patch("mlx_websockets.cli.find_all_mlx_processes")
    @patch("mlx_websockets.cli.stop_background_server")
    def test_background_stop_command_success(self, mock_stop, mock_find_all):
        """Test successful background stop command."""
        # Mock some processes running
        mock_find_all.return_value = [
            ProcessInfo(pid=12345, port=8765, start_time=123.0, is_daemon=True)
        ]
        mock_stop.return_value = True

        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            main(["background", "stop"])

        mock_stop.assert_called_once()
        output = mock_stdout.getvalue()
        assert "MLX server(s) stopped successfully" in output

    @patch("mlx_websockets.cli.find_all_mlx_processes")
    @patch("mlx_websockets.cli.stop_background_server")
    def test_background_stop_command_failure(self, mock_stop, mock_find_all):
        """Test failed background stop command."""
        # Mock some processes running
        mock_find_all.return_value = [
            ProcessInfo(pid=12345, port=8765, start_time=123.0, is_daemon=True)
        ]
        mock_stop.return_value = False

        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            main(["background", "stop"])

        mock_stop.assert_called_once()
        output = mock_stdout.getvalue()
        assert "Failed to stop some MLX servers" in output

    def test_background_no_subcommand_shows_help(self, capsys):
        """Test that background without subcommand shows help."""
        with pytest.raises(SystemExit) as exc_info:
            main(["background"])
        # Exit code is 0 when showing help
        assert exc_info.value.code == 0

        captured = capsys.readouterr()
        # Help text for background subcommand
        assert "usage: mlx background" in captured.out
        assert "Background commands" in captured.out

    @patch("mlx_websockets.cli.find_all_mlx_processes")
    def test_status_command_running(self, mock_find_all):
        """Test status command when server is running."""
        # Mock a running process
        mock_find_all.return_value = [
            ProcessInfo(pid=12345, port=8765, start_time=123.0, is_daemon=True, model="test-model")
        ]

        with patch("mlx_websockets.cli.get_server_status") as mock_status:
            mock_status.return_value = {
                "pid": 12345,
                "port": 8765,
                "started": "2024-01-01 12:00:00",
                "model": "test-model",
            }

            with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
                main(["status"])

            output = mock_stdout.getvalue()
            assert "Found 1 MLX server(s) running:" in output
            assert "Model: test-model" in output
            assert "Port: 8765" in output
            assert "PID: 12345" in output

    @patch("mlx_websockets.daemon.find_all_mlx_processes")
    def test_status_command_not_running(self, mock_find_all):
        """Test status command when server is not running."""
        mock_find_all.return_value = []

        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            main(["status"])

        output = mock_stdout.getvalue()
        assert "No MLX servers are running" in output

    def test_port_type_conversion(self):
        """Test that port numbers are correctly converted to strings."""
        with patch("mlx_websockets.daemon.find_all_mlx_processes", return_value=[]):
            with patch("mlx_websockets.cli.run_server") as mock_run_server:
                main(["serve", "--port", "8080"])

                mock_run_server.assert_called_once()
                assert "--port" in sys.argv
                assert "8080" in sys.argv

    def test_boolean_flags(self):
        """Test that boolean flags work correctly."""
        with patch("mlx_websockets.daemon.find_all_mlx_processes", return_value=[]):
            with patch("mlx_websockets.cli.run_server") as mock_run_server:
                main(["serve", "--trust-remote-code"])

                mock_run_server.assert_called_once()
                assert "--trust-remote-code" in sys.argv

    @patch("mlx_websockets.cli.start_background_server")
    @patch("mlx_websockets.cli.get_server_status")
    def test_background_serve_removes_none_values(self, mock_status, mock_start):
        """Test that None values are removed from background serve args."""
        mock_start.return_value = None  # start_background_server returns None
        mock_status.return_value = {"port": 8765}

        main(["background", "serve"])

        mock_start.assert_called_once()
        call_args = mock_start.call_args[0][0]  # Get the dictionary argument
        # Should not contain any None values
        assert None not in call_args.values()
        # Should have default values
        assert call_args.get("model") == "mlx-community/gemma-3-4b-it-4bit"
        assert call_args.get("port") == 8765

    def test_temperature_float_conversion(self):
        """Test that temperature is correctly handled as float."""
        with patch("mlx_websockets.daemon.find_all_mlx_processes", return_value=[]):
            with patch("mlx_websockets.cli.run_server") as mock_run_server:
                main(["serve", "--temperature", "0.9"])

                mock_run_server.assert_called_once()
                assert "--temperature" in sys.argv
                assert "0.9" in sys.argv

    def test_seed_int_conversion(self):
        """Test that seed is correctly handled as int."""
        with patch("mlx_websockets.daemon.find_all_mlx_processes", return_value=[]):
            with patch("mlx_websockets.cli.run_server") as mock_run_server:
                main(["serve", "--seed", "12345"])

                mock_run_server.assert_called_once()
                assert "--seed" in sys.argv
                assert "12345" in sys.argv
