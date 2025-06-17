"""Background daemon management for MLX WebSocket server."""

import json
import os
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import psutil

from .exceptions import DaemonError
from .logging_utils import get_logger, log_info
from .process_registry import ProcessInfo, get_registry

logger = get_logger(__name__)

try:
    from rich.console import Console
    from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn

    RICH_AVAILABLE = True
    console: Optional[Console] = Console()
except ImportError:
    RICH_AVAILABLE = False
    console = None


def get_config_dir() -> Path:
    """Get the configuration directory for MLX WebSockets."""
    if sys.platform == "linux":
        # Linux: use XDG config directory
        config_dir = Path.home() / ".config" / "mlx-websockets"
    elif sys.platform == "win32":
        # Windows: use APPDATA directory
        appdata = os.environ.get("APPDATA")
        if appdata:
            config_dir = Path(appdata) / "mlx-websockets"
        else:
            config_dir = Path.home() / "mlx-websockets"
    else:
        # macOS and others: use hidden directory in home
        config_dir = Path.home() / ".mlx-websockets"

    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir


def get_pid_file() -> Path:
    """Get the PID file path."""
    return get_config_dir() / "mlx-server.pid"


def get_log_file() -> Path:
    """Get the log file path."""
    return get_config_dir() / "mlx-server.log"


def get_config_file() -> Path:
    """Get the config file path."""
    return get_config_dir() / "mlx-server.json"


def find_all_mlx_processes() -> list[ProcessInfo]:
    """Find ALL MLX server processes, not just daemon-managed ones."""
    registry = get_registry()
    registered_processes = []

    for reg_proc in registry.get_all_processes():
        if is_process_running(reg_proc.pid):
            registered_processes.append(
                ProcessInfo(
                    pid=reg_proc.pid,
                    port=reg_proc.port,
                    start_time=reg_proc.start_time,
                    is_daemon=is_our_daemon_process(reg_proc.pid) or reg_proc.is_daemon,
                    cmdline=[],
                    model=reg_proc.model,
                )
            )

    if not psutil:
        logger.warning("psutil not available, cannot find unregistered MLX processes")
        return registered_processes

    registered_pids = {p.pid for p in registered_processes}

    for proc in psutil.process_iter(["pid", "name", "cmdline", "create_time"]):
        try:
            pid = proc.info["pid"]
            if pid in registered_pids:
                continue

            cmdline = proc.info.get("cmdline") or []
            cmdline_str = " ".join(cmdline).lower()

            # Check for any MLX-related process
            mlx_patterns = [
                "mlx serve",
                "mlx_websockets",
                "mlx-websockets",
                "mlx_websockets.server",
                "mlx_websockets.cli",
            ]

            if any(pattern in cmdline_str for pattern in mlx_patterns):
                # Extract port from command line
                port = 8765  # default
                model = ""

                for i, arg in enumerate(cmdline):
                    if arg in ["--port", "-p"] and i + 1 < len(cmdline):
                        try:
                            port = int(cmdline[i + 1])
                        except (ValueError, IndexError):
                            pass
                    elif arg in ["--model", "-m"] and i + 1 < len(cmdline):
                        model = cmdline[i + 1]

                # Try to get port from network connections if not found
                if port == 8765:
                    try:
                        connections = proc.net_connections(kind="inet")
                        for conn in connections:
                            if conn.status == "LISTEN" and 8765 <= conn.laddr.port <= 8775:
                                port = conn.laddr.port
                                break
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass

                # Double-check the process is still running before adding
                if is_process_running(pid):
                    # Verify port is actually in use by checking if we can connect
                    if is_port_in_use(port):
                        registered_processes.append(
                            ProcessInfo(
                                pid=pid,
                                port=port,
                                start_time=proc.info.get("create_time", time.time()),
                                is_daemon=is_our_daemon_process(pid),
                                cmdline=cmdline,
                                model=model,
                            )
                        )
        except (psutil.NoSuchProcess, psutil.AccessDenied, AttributeError):
            continue

    return registered_processes


def is_any_mlx_server_running() -> tuple[bool, Optional[ProcessInfo]]:
    """Check if ANY MLX server is running (daemon-managed or not)."""
    processes = find_all_mlx_processes()
    if processes:
        # Return the first one found
        return True, processes[0]
    return False, None


def is_our_daemon_process(pid: int) -> bool:
    """Check if a PID belongs to our daemon-managed process."""
    try:
        pid_file = get_pid_file()
        if pid_file.exists():
            saved_pid = int(pid_file.read_text().strip())
            return pid == saved_pid
    except (OSError, ValueError):
        pass
    return False


def is_port_in_use(port: int, host: str = "0.0.0.0") -> bool:
    """Check if a port is in use."""
    sock = None
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex((host, port))
        return result == 0
    except OSError:
        return False
    finally:
        if sock:
            sock.close()


def format_time(timestamp: Optional[float]) -> str:
    """Format a timestamp to a readable string."""
    if timestamp is None:
        return "Unknown"
    try:
        dt = datetime.fromtimestamp(timestamp)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, OSError):
        return "Unknown"


def format_runtime(start_time: Optional[float]) -> str:
    """Format runtime duration in a human-readable format."""
    if start_time is None:
        return "Unknown"
    try:
        elapsed = time.time() - start_time
        if elapsed < 60:
            return f"{int(elapsed)} seconds"
        elif elapsed < 3600:
            minutes = int(elapsed / 60)
            seconds = int(elapsed % 60)
            return f"{minutes} min {seconds} sec"
        else:
            hours = int(elapsed / 3600)
            minutes = int((elapsed % 3600) / 60)
            if hours == 1:
                return f"1 hour {minutes} min"
            else:
                return f"{hours} hours {minutes} min"
    except (ValueError, TypeError):
        return "Unknown"


def find_next_available_port() -> int:
    """Find the next available port starting from 8766."""
    return find_available_port(8766)


def find_available_port(start_port: int = 8765, max_attempts: int = 100) -> int:
    """Find an available port starting from the given port."""
    for i in range(max_attempts):
        port = start_port + i
        sock = None
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("", port))
            # Successfully bound, close socket and return port
            sock.close()
            return port
        except OSError:
            # Port is in use or not available, try next one
            continue
        except Exception:
            # Unexpected error, try next port
            continue
    raise RuntimeError(
        f"Could not find available port after {max_attempts} attempts starting from {start_port}"
    )


def is_process_running(pid: int) -> bool:
    """Check if a process with the given PID is running."""
    try:
        # First try using os.kill(pid, 0) - fastest method
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        # Process doesn't exist
        return False
    except PermissionError:
        # Process exists but we don't have permission to signal it
        return True
    except OSError:
        # Other OS errors - assume process doesn't exist
        return False


def get_server_status() -> Optional[dict[str, Any]]:
    """Get the status of the background server."""
    pid_file = get_pid_file()
    config_file = get_config_file()

    if not pid_file.exists() or not config_file.exists():
        return None

    try:
        pid = int(pid_file.read_text().strip())
        config = json.loads(config_file.read_text())

        if not is_process_running(pid):
            # Clean up stale files
            pid_file.unlink()
            config_file.unlink()
            return None

        return {
            "pid": pid,
            "port": config.get("port", 8765),
            "started": config.get("started", "Unknown"),
            "model": config.get("model", "Unknown"),
        }
    except (ValueError, json.JSONDecodeError):
        return None


def start_background_server(server_config: dict[str, Any]) -> None:
    """Start the WebSocket server in the background, handling any existing MLX processes."""

    # Step 1: Check for ANY MLX process
    running, existing_process = is_any_mlx_server_running()

    if running and existing_process is not None:
        if existing_process.is_daemon:
            # It's our daemon - give proper error
            status = get_server_status()
            raise DaemonError(
                f"MLX server already running (daemon-managed):\n"
                f"  Model: {status.get('model', existing_process.model or 'unknown') if status else existing_process.model or 'unknown'}\n"
                f"  Port: {status.get('port', existing_process.port) if status else existing_process.port}\n"
                f"  PID: {existing_process.pid}\n\n"
                f"To stop it: mlx bg stop\n"
                f"To see status: mlx status"
            )
        else:
            # External process - offer to take it over
            model_info = f"  Model: {existing_process.model}\n" if existing_process.model else ""
            raise DaemonError(
                f"MLX server already running (not daemon-managed):\n"
                f"{model_info}"
                f"  Port: {existing_process.port}\n"
                f"  PID: {existing_process.pid}\n"
                f"  Running since: {format_time(existing_process.start_time)}\n"
                f"  Running for: {format_runtime(existing_process.start_time)}\n\n"
                f"Options:\n"
                f"  1. Stop it first: mlx bg stop\n"
                f"  2. Use a different port: mlx bg serve --port {find_next_available_port()}"
            )

    # Step 2: Verify port is actually free
    port = server_config.get("port", 8765)
    if is_port_in_use(port):
        # Port is in use by non-MLX process
        new_port = find_available_port(port)
        if new_port != port:
            log_info(f"Port {port} is busy, using {new_port} instead")
            server_config["port"] = new_port

    # Extract configuration with automatic port assignment done above
    port = server_config.get("port", 8765)

    # Prepare command
    cmd = [sys.executable, "-m", "mlx_websockets.server"]

    # Add arguments from server_config
    if "model" in server_config:
        cmd.extend(["--model", server_config["model"]])
    if "host" in server_config:
        cmd.extend(["--host", server_config["host"]])
    if "port" in server_config:
        cmd.extend(["--port", str(server_config["port"])])
    if server_config.get("trust_remote_code"):
        cmd.append("--trust-remote-code")
    if "tokenizer_config" in server_config:
        cmd.extend(["--tokenizer-config", server_config["tokenizer_config"]])
    if "chat_template" in server_config:
        cmd.extend(["--chat-template", server_config["chat_template"]])
    if "max_tokens" in server_config:
        cmd.extend(["--max-tokens", str(server_config["max_tokens"])])
    if "temperature" in server_config:
        cmd.extend(["--temperature", str(server_config["temperature"])])
    if "seed" in server_config:
        cmd.extend(["--seed", str(server_config["seed"])])

    # Start the process
    log_file = get_log_file()
    with open(log_file, "w") as log:
        process = subprocess.Popen(
            cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,  # Detach from parent process
        )

    # Wait a moment to ensure process starts
    time.sleep(2)

    # Check if process is still running
    if not is_process_running(process.pid):
        raise RuntimeError(
            "Failed to start background server. Check the log file at: " + str(log_file)
        )

    # Save PID and config
    get_pid_file().write_text(str(process.pid))
    config = {
        "port": port,
        "started": datetime.now().isoformat(),
        "model": server_config.get("model", "mlx-community/gemma-3-4b-it-4bit"),
        "command": cmd,
    }
    get_config_file().write_text(json.dumps(config, indent=2))

    # Register with process registry
    registry = get_registry()
    registry.register_process(
        ProcessInfo(
            pid=process.pid,
            port=port,
            model=server_config.get("model", "mlx-community/gemma-3-4b-it-4bit"),
            start_time=time.time(),
            is_daemon=True,
        )
    )


def stop_background_server(daemon_only: bool = False) -> bool:
    """Stop MLX WebSocket servers.

    Args:
        daemon_only: If True, stop only daemon-managed servers.
                    If False (default), stop ALL MLX servers.

    Returns:
        True if any server was stopped, False otherwise
    """
    # First try to stop daemon-managed server
    status = get_server_status()
    if status:
        pid = status["pid"]
        try:
            # Send SIGTERM for graceful shutdown
            os.kill(pid, signal.SIGTERM)

            # Wait for process to terminate
            for _ in range(10):
                if not is_process_running(pid):
                    break
                time.sleep(0.5)

            # Force kill if still running
            if is_process_running(pid):
                os.kill(pid, signal.SIGKILL)
                time.sleep(0.5)

            # Clean up files
            get_pid_file().unlink(missing_ok=True)
            get_config_file().unlink(missing_ok=True)

            # Unregister from process registry
            registry = get_registry()
            registry.unregister_process(pid)

            return True
        except (OSError, subprocess.CalledProcessError) as e:
            logger.error(f"Failed to stop daemon: {e}")
            # Continue to check for other processes
        except ValueError as e:
            logger.error(f"Invalid PID in status file: {e}")
            # Continue to check for other processes

    # If not daemon_only, stop ALL MLX processes
    if not daemon_only:
        all_processes = find_all_mlx_processes()
        stopped_count = 0

        # Filter out already stopped daemon process
        if status and not is_process_running(status["pid"]):
            all_processes = [p for p in all_processes if p.pid != status["pid"]]

        # Create progress bar if Rich is available and we have processes to stop
        if RICH_AVAILABLE and all_processes:
            with Progress(
                SpinnerColumn(style="purple"),
                TextColumn("[cream]Stopping MLX process[/cream]"),
                BarColumn(complete_style="green", finished_style="green"),
                TextColumn("[purple]{task.fields[status]}[/purple]"),
                console=console,
                transient=True,
            ) as progress:
                task = progress.add_task("Stopping processes", total=len(all_processes), status="")

                for process in all_processes:
                    if process.is_daemon and status:
                        # Already handled above
                        progress.advance(task)
                        continue

                    try:
                        # Update progress status
                        progress.update(task, status=f"PID {process.pid} (Port {process.port})")

                        # Send SIGTERM
                        os.kill(process.pid, signal.SIGTERM)

                        # Wait for termination
                        terminated = False
                        for _ in range(10):
                            if not is_process_running(process.pid):
                                terminated = True
                                break
                            time.sleep(0.5)

                        # Force kill if needed
                        if not terminated and is_process_running(process.pid):
                            os.kill(process.pid, signal.SIGKILL)
                            time.sleep(0.5)
                            terminated = not is_process_running(process.pid)

                        if terminated:
                            stopped_count += 1
                            # Try to unregister from registry
                            registry = get_registry()
                            registry.unregister_process(process.pid)

                        # Advance progress
                        progress.advance(task)

                    except (OSError, ProcessLookupError) as e:
                        if "No such process" not in str(e):
                            logger.error(f"Failed to stop process {process.pid}: {e}")
                        progress.advance(task)
        else:
            # Fallback without progress bar
            for process in all_processes:
                if process.is_daemon and status:
                    # Already handled above
                    continue

                try:
                    logger.info(f"Stopping MLX process (PID: {process.pid}, Port: {process.port})")

                    # Send SIGTERM
                    os.kill(process.pid, signal.SIGTERM)

                    # Wait for termination
                    terminated = False
                    for _ in range(10):
                        if not is_process_running(process.pid):
                            terminated = True
                            break
                        time.sleep(0.5)

                    # Force kill if needed
                    if not terminated and is_process_running(process.pid):
                        os.kill(process.pid, signal.SIGKILL)
                        time.sleep(0.5)
                        terminated = not is_process_running(process.pid)

                    if terminated:
                        stopped_count += 1
                        # Try to unregister from registry
                        registry = get_registry()
                        registry.unregister_process(process.pid)

                except (OSError, ProcessLookupError) as e:
                    if "No such process" not in str(e):
                        logger.error(f"Failed to stop process {process.pid}: {e}")

        return stopped_count > 0 or (status is not None)

    return status is not None
