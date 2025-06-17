"""Resource monitoring utilities for mlx-websockets."""

import logging
import platform
import subprocess
import threading
import time
from typing import Any, Optional

import psutil

try:
    import mlx.core as mx
except ImportError:
    mx = None  # type: ignore

from .logging_utils import log_error, log_warning

logger = logging.getLogger(__name__)


class ResourceMonitor:
    """Monitor system resources and provide warnings."""

    def __init__(
        self,
        low_memory_threshold: float = 0.15,  # 15% free memory
        critical_memory_threshold: float = 0.05,  # 5% free memory
        check_interval: float = 30.0,  # Check every 30 seconds
        enable_battery_monitoring: bool = True,
    ):
        self.low_memory_threshold = low_memory_threshold
        self.critical_memory_threshold = critical_memory_threshold
        self.check_interval = check_interval
        self.enable_battery_monitoring = enable_battery_monitoring

        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._memory_warned = False
        self._battery_warned = False
        self._last_memory_check = 0.0
        self._last_battery_check = 0.0

    def start(self) -> None:
        """Start resource monitoring in a background thread."""
        if self._monitoring:
            return

        self._monitoring = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop, daemon=True, name="ResourceMonitor"
        )
        self._monitor_thread.start()
        logger.debug("Resource monitoring started")

    def stop(self) -> None:
        """Stop resource monitoring."""
        self._monitoring = False
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=1.0)
        logger.debug("Resource monitoring stopped")

    def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self._monitoring:
            try:
                # Check memory
                self._check_memory()

                # Check battery (less frequently)
                if self.enable_battery_monitoring and time.time() - self._last_battery_check > 60:
                    self._check_battery()
                    self._last_battery_check = time.time()

                # Sleep until next check
                time.sleep(self.check_interval)

            except Exception as e:
                logger.error(f"Error in resource monitor: {e}", exc_info=True)
                time.sleep(self.check_interval)

    def _check_memory(self) -> None:
        """Check memory usage and warn if low."""
        try:
            # System memory
            mem = psutil.virtual_memory()
            free_percent = mem.available / mem.total

            # MLX memory if available
            mlx_memory_gb = 0.0
            if mx is not None:
                try:
                    mlx_memory_gb = mx.get_active_memory() / (1024**3)
                except Exception:
                    pass

            # Check thresholds
            if free_percent < self.critical_memory_threshold:
                if not self._memory_warned:
                    msg = f"CRITICAL: System memory very low ({free_percent * 100:.1f}% free)"
                    if mlx_memory_gb > 0:
                        msg += f", MLX using {mlx_memory_gb:.1f}GB"
                    log_error(msg, logger)
                    self._memory_warned = True

            elif free_percent < self.low_memory_threshold:
                if not self._memory_warned:
                    msg = f"WARNING: System memory low ({free_percent * 100:.1f}% free)"
                    if mlx_memory_gb > 0:
                        msg += f", MLX using {mlx_memory_gb:.1f}GB"
                    log_warning(msg, logger)
                    self._memory_warned = True
            else:
                self._memory_warned = False

        except Exception as e:
            logger.debug(f"Failed to check memory: {e}")

    def _check_battery(self) -> None:
        """Check battery status and warn if low."""
        if platform.system() != "Darwin":
            return

        try:
            battery = psutil.sensors_battery()
            if battery is None:
                return

            # Check if on battery power and low
            if not battery.power_plugged and battery.percent < 20:
                if not self._battery_warned:
                    log_warning(
                        f"Running on battery power ({battery.percent}%). "
                        "Consider plugging in for optimal performance.",
                        logger,
                    )
                    self._battery_warned = True
            else:
                self._battery_warned = False

        except Exception as e:
            logger.debug(f"Failed to check battery: {e}")

    def get_resource_stats(self) -> dict[str, Any]:
        """Get current resource statistics."""
        stats: dict[str, Any] = {}

        try:
            # Memory stats
            mem = psutil.virtual_memory()
            stats["memory"] = {
                "total_gb": mem.total / (1024**3),
                "available_gb": mem.available / (1024**3),
                "percent_used": mem.percent,
                "percent_free": (mem.available / mem.total) * 100,
            }

            # MLX memory
            if mx is not None:
                try:
                    stats["mlx_memory_gb"] = mx.get_active_memory() / (1024**3)
                except Exception:
                    stats["mlx_memory_gb"] = None

            # CPU stats
            stats["cpu"] = {
                "percent": psutil.cpu_percent(interval=0.1),
                "count": psutil.cpu_count(),
            }

            # Battery stats (macOS)
            if platform.system() == "Darwin":
                battery = psutil.sensors_battery()
                if battery:
                    stats["battery"] = {
                        "percent": battery.percent,
                        "plugged": battery.power_plugged,
                        "time_remaining": battery.secsleft if battery.secsleft != -1 else None,
                    }

        except Exception as e:
            logger.error(f"Failed to get resource stats: {e}")

        return stats


def check_memory_available(required_gb: float = 2.0) -> tuple[bool, str]:
    """
    Check if enough memory is available.

    Args:
        required_gb: Required memory in GB

    Returns:
        Tuple of (is_available, message)
    """
    try:
        mem = psutil.virtual_memory()
        available_gb = mem.available / (1024**3)

        if available_gb < required_gb:
            return (
                False,
                f"Insufficient memory: {available_gb:.1f}GB available, {required_gb:.1f}GB required",
            )

        return True, f"Memory check passed: {available_gb:.1f}GB available"

    except Exception as e:
        logger.error(f"Failed to check memory: {e}")
        return True, "Could not check memory"


def get_thermal_state() -> Optional[str]:
    """
    Get thermal state on macOS.

    Returns:
        Thermal state string or None
    """
    if platform.system() != "Darwin":
        return None

    try:
        # Use powermetrics to check thermal state
        result = subprocess.run(["pmset", "-g", "therm"], capture_output=True, text=True, timeout=1)

        if result.returncode == 0:
            output = result.stdout.strip()
            # Parse thermal state from output
            if "CPU_Speed_Limit" in output:
                return "throttled"
            else:
                return "normal"

    except Exception as e:
        logger.debug(f"Failed to check thermal state: {e}")

    return None


def suggest_memory_optimizations(mlx_memory_gb: float) -> list[str]:
    """
    Suggest memory optimizations based on current usage.

    Args:
        mlx_memory_gb: Current MLX memory usage in GB

    Returns:
        List of optimization suggestions
    """
    suggestions = []

    if mlx_memory_gb > 8:
        suggestions.append("Consider using a smaller model or quantized version")
        suggestions.append("Try reducing max_tokens to limit memory usage")

    if mlx_memory_gb > 4:
        suggestions.append("Close other applications to free up memory")
        suggestions.append("Consider using MLX memory mapping if available")

    mem = psutil.virtual_memory()
    if mem.percent > 90:
        suggestions.append("System memory critically low - restart may help")
        suggestions.append("Consider upgrading system RAM for better performance")

    return suggestions
