"""MLX WebSockets - A streaming server for MLX models."""

try:
    from ._version import __version__
except ImportError:
    # Package is not installed, or setuptools_scm failed
    __version__ = "0.0.0+unknown"

# Expose submodules for backward compatibility
from . import cli, daemon, process_registry, resource_monitor, server

__all__ = ["__version__", "cli", "daemon", "server", "process_registry", "resource_monitor"]
