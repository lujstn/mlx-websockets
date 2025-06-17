"""Process registry for tracking all MLX WebSocket server instances."""

import json
import socket
import time
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import psutil

from .logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class ProcessInfo:
    """Information about a running MLX process."""

    pid: int
    port: int
    model: Optional[str] = None
    start_time: Optional[float] = None
    is_daemon: bool = False
    cmdline: Optional[list[str]] = None

    def __post_init__(self):
        if self.start_time is None:
            self.start_time = time.time()

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        # Don't serialize cmdline as it's only for runtime use
        return {
            "pid": self.pid,
            "port": self.port,
            "model": self.model,
            "start_time": self.start_time,
            "is_daemon": self.is_daemon,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ProcessInfo":
        """Create from dictionary."""
        return cls(
            pid=data["pid"],
            port=data["port"],
            model=data.get("model"),
            start_time=data.get("start_time", time.time()),
            is_daemon=data.get("is_daemon", False),
        )


class ProcessRegistry:
    """Registry for tracking all MLX WebSocket server processes."""

    def __init__(self):
        self.registry_dir = Path.home() / ".mlx-websockets" / "processes"
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self._clean_stale_entries()

    def _get_process_file(self, pid: int) -> Path:
        """Get the file path for a process."""
        return self.registry_dir / f"{pid}.json"

    def _is_process_running(self, pid: int) -> bool:
        """Check if a process is still running."""
        try:
            process = psutil.Process(pid)
            return process.is_running()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return False

    def _is_port_in_use(self, port: int) -> bool:
        """Check if a port is in use."""
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
            try:
                sock.bind(("", port))
                return False
            except OSError:
                return True

    def _clean_stale_entries(self):
        """Remove entries for processes that are no longer running."""
        for file_path in self.registry_dir.glob("*.json"):
            try:
                pid = int(file_path.stem)
                if not self._is_process_running(pid):
                    file_path.unlink()
                    logger.debug(f"Removed stale entry for PID {pid}")
            except (ValueError, OSError) as e:
                logger.error(f"Error cleaning stale entry {file_path}: {e}")

    def register_process(self, process_info: ProcessInfo) -> bool:
        """Register a new MLX process."""
        try:
            # Check if process is already registered
            if self._get_process_file(process_info.pid).exists():
                logger.warning(f"Process {process_info.pid} already registered")
                return False

            # Write process info
            file_path = self._get_process_file(process_info.pid)
            with open(file_path, "w") as f:
                json.dump(process_info.to_dict(), f, indent=2)

            logger.info(f"Registered process {process_info.pid} on port {process_info.port}")
            return True
        except Exception as e:
            logger.error(f"Failed to register process {process_info.pid}: {e}")
            return False

    def unregister_process(self, pid: int) -> bool:
        """Unregister an MLX process."""
        try:
            file_path = self._get_process_file(pid)
            if file_path.exists():
                file_path.unlink()
                logger.debug(f"Unregistered process {pid}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to unregister process {pid}: {e}")
            return False

    def get_all_processes(self) -> list[ProcessInfo]:
        """Get all registered MLX processes."""
        processes = []
        self._clean_stale_entries()

        for file_path in self.registry_dir.glob("*.json"):
            try:
                with open(file_path) as f:
                    data = json.load(f)
                    process_info = ProcessInfo.from_dict(data)

                    # Verify process is still running
                    if self._is_process_running(process_info.pid):
                        # Additional validation: check if it's really an MLX process
                        if self._is_mlx_process(process_info.pid):
                            processes.append(process_info)
                        else:
                            # Not an MLX process anymore, remove entry
                            logger.warning(
                                f"Process {process_info.pid} is not an MLX process, removing entry"
                            )
                            file_path.unlink()
                    else:
                        file_path.unlink()
            except Exception as e:
                logger.error(f"Error reading process file {file_path}: {e}")
                # Remove corrupted file
                try:
                    file_path.unlink()
                except OSError:
                    pass

        return processes

    def find_process_on_port(self, port: int) -> Optional[ProcessInfo]:
        """Find an MLX process running on a specific port."""
        for process in self.get_all_processes():
            if process.port == port:
                return process
        return None

    def find_mlx_process_ports(self) -> list[int]:
        """Get all ports used by MLX processes."""
        return [p.port for p in self.get_all_processes()]

    def is_mlx_port(self, port: int) -> bool:
        """Check if a port is used by an MLX process."""
        return self.find_process_on_port(port) is not None

    def _is_mlx_process(self, pid: int) -> bool:
        """Check if a process is actually an MLX process."""
        try:
            process = psutil.Process(pid)
            cmdline = " ".join(process.cmdline()).lower()

            # Check for MLX-related patterns
            mlx_patterns = [
                "mlx serve",
                "mlx_websockets",
                "mlx-websockets",
                "mlx_websockets.server",
                "mlx_websockets.cli",
            ]

            return any(pattern in cmdline for pattern in mlx_patterns)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return False

    def detect_unregistered_mlx_processes(self) -> list[tuple[int, int]]:
        """Detect MLX processes that aren't in the registry.

        Returns:
            List of (pid, port) tuples for unregistered MLX processes.
        """
        unregistered = []
        registered_pids = {p.pid for p in self.get_all_processes()}

        # Check all processes for MLX-related command lines
        for proc in psutil.process_iter(["pid", "name", "cmdline"]):
            try:
                pid = proc.info["pid"]
                if pid in registered_pids:
                    continue

                cmdline_list = proc.info.get("cmdline") or []
                cmdline = " ".join(cmdline_list)

                # Check for MLX-related processes
                if any(
                    pattern in cmdline
                    for pattern in [
                        "mlx serve",
                        "mlx_websockets",
                        "mlx-websockets",
                        "mlx_websockets.server",
                        "mlx_websockets.cli",
                    ]
                ):
                    # Try to find the port
                    connections = proc.net_connections(kind="inet")
                    for conn in connections:
                        if conn.status == "LISTEN" and conn.laddr.port >= 8765:
                            unregistered.append((pid, conn.laddr.port))
                            break
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        return unregistered


# Global registry instance
_registry = None


def get_registry() -> ProcessRegistry:
    """Get the global process registry instance."""
    global _registry
    if _registry is None:
        _registry = ProcessRegistry()
    return _registry
