"""Tests for process registry functionality."""

import os
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from mlx_websockets.process_registry import ProcessInfo, ProcessRegistry


class TestProcessRegistry:
    """Test process registry functionality."""

    @pytest.fixture
    def temp_registry(self):
        """Create a temporary registry for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ProcessRegistry()
            # Override the registry directory
            registry.registry_dir = Path(tmpdir) / "processes"
            registry.registry_dir.mkdir(parents=True, exist_ok=True)
            yield registry

    @patch.object(ProcessRegistry, "_is_mlx_process", return_value=True)
    def test_register_process(self, mock_is_mlx, temp_registry):
        """Test registering a process."""
        process_info = ProcessInfo(
            pid=os.getpid(), port=8765, model="test-model", start_time=time.time(), is_daemon=False
        )

        # Register the process
        assert temp_registry.register_process(process_info)

        # Verify it's registered
        processes = temp_registry.get_all_processes()
        assert len(processes) == 1
        assert processes[0].pid == os.getpid()
        assert processes[0].port == 8765
        assert processes[0].model == "test-model"

    def test_unregister_process(self, temp_registry):
        """Test unregistering a process."""
        process_info = ProcessInfo(pid=os.getpid(), port=8765, model="test-model")

        # Register and then unregister
        temp_registry.register_process(process_info)
        assert temp_registry.unregister_process(os.getpid())

        # Verify it's gone
        processes = temp_registry.get_all_processes()
        assert len(processes) == 0

    @patch.object(ProcessRegistry, "_is_mlx_process", return_value=True)
    def test_find_process_on_port(self, mock_is_mlx, temp_registry):
        """Test finding a process by port."""
        process_info = ProcessInfo(pid=os.getpid(), port=8765, model="test-model")

        temp_registry.register_process(process_info)

        # Find by port
        found = temp_registry.find_process_on_port(8765)
        assert found is not None
        assert found.pid == os.getpid()

        # Non-existent port
        not_found = temp_registry.find_process_on_port(9999)
        assert not_found is None

    @patch.object(ProcessRegistry, "_is_mlx_process", return_value=True)
    def test_find_mlx_process_ports(self, mock_is_mlx, temp_registry):
        """Test finding all MLX process ports."""
        # Register only this process (which is actually running)
        process_info = ProcessInfo(pid=os.getpid(), port=8765, model="test-model")
        temp_registry.register_process(process_info)

        ports = temp_registry.find_mlx_process_ports()
        assert len(ports) == 1
        assert ports[0] == 8765

    @patch.object(ProcessRegistry, "_is_mlx_process", return_value=True)
    def test_is_mlx_port(self, mock_is_mlx, temp_registry):
        """Test checking if a port is used by MLX."""
        process_info = ProcessInfo(pid=os.getpid(), port=8765, model="test-model")

        temp_registry.register_process(process_info)

        assert temp_registry.is_mlx_port(8765)
        assert not temp_registry.is_mlx_port(9999)

    def test_clean_stale_entries(self, temp_registry):
        """Test cleaning stale entries."""
        # Register a process with a fake PID that doesn't exist
        fake_pid = 99999  # Very unlikely to exist
        process_info = ProcessInfo(pid=fake_pid, port=8765, model="test-model")

        # Manually create the file
        file_path = temp_registry._get_process_file(fake_pid)
        with open(file_path, "w") as f:
            import json

            json.dump(process_info.to_dict(), f)

        # Clean stale entries
        temp_registry._clean_stale_entries()

        # Verify it's gone
        assert not file_path.exists()

    def test_process_info_serialization(self):
        """Test ProcessInfo serialization."""
        process_info = ProcessInfo(
            pid=12345, port=8765, model="test-model", start_time=1234567890.0, is_daemon=True
        )

        # To dict
        data = process_info.to_dict()
        assert data["pid"] == 12345
        assert data["port"] == 8765
        assert data["model"] == "test-model"
        assert data["start_time"] == 1234567890.0
        assert data["is_daemon"] is True

        # From dict
        restored = ProcessInfo.from_dict(data)
        assert restored.pid == process_info.pid
        assert restored.port == process_info.port
        assert restored.model == process_info.model
        assert restored.start_time == process_info.start_time
        assert restored.is_daemon == process_info.is_daemon
