"""Logging utilities for mlx-websockets."""

import json
import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Any, Optional

# Try to import rich for enhanced console output
try:
    from rich.console import Console
    from rich.logging import RichHandler
    from rich.theme import Theme

    RICH_AVAILABLE = True

    # Custom color theme matching server.py
    custom_theme = Theme(
        {
            "white": "#fffbeb",
            "cream": "#faf6e6",
            "dim_cream": "#E3DDDD",
            "pink": "#ff79c6",
            "purple": "#a07ae1",
            "bold_purple": "bold #a07ae1",
            "yellow": "#ffd230",
            "bold_yellow": "bold #ffd230",
            "cyan": "#00d3f2",
            "green": "#50fa7b",
            "red": "#ff5555",
            "dim": "#a07ae1",
            "bold_white": "bold #fffbeb",
        }
    )

    console = Console(theme=custom_theme)
except ImportError:
    RICH_AVAILABLE = False
    console = None  # type: ignore


# ANSI color constants matching the Rich theme
ANSI_COLORS = {
    "cream": "\033[97m",  # #faf6e6 approximated
    "dim_cream": "\033[90m",  # #E3DDDD approximated
    "purple": "\033[35m",  # #a07ae1 approximated
    "amber": "\033[33m",  # #ffd230 approximated
    "yellow": "\033[33m",  # #ffd230 approximated
    "red": "\033[31m",  # #ff5555 approximated
    "cyan": "\033[36m",  # #00d3f2 approximated
    "green": "\033[32m",  # #50fa7b approximated
    "reset": "\033[0m",
    "dim": "\033[35m",  # Purple for dim
}


class ANSIColorHandler(logging.StreamHandler):
    """Custom logging handler with ANSI color support."""

    COLORS = {
        logging.DEBUG: ANSI_COLORS["dim"],  # Dim (purple)
        logging.INFO: ANSI_COLORS["purple"],  # Purple
        logging.WARNING: ANSI_COLORS["amber"],  # Amber
        logging.ERROR: ANSI_COLORS["red"],  # Red
        logging.CRITICAL: ANSI_COLORS["red"],  # Red
    }
    RESET = ANSI_COLORS["reset"]

    def __init__(self, stream=None):
        super().__init__(stream or sys.stdout)

    def emit(self, record):
        try:
            msg = self.format(record)
            color = self.COLORS.get(record.levelno, self.RESET)
            self.stream.write(f"{color}{msg}{self.RESET}\n")
            self.flush()
        except Exception:
            self.handleError(record)


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        log_obj: dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
        # Add custom fields if present
        for attr in ["client_id", "request_id", "duration", "status"]:
            if hasattr(record, attr):
                log_obj[attr] = getattr(record, attr)
        return json.dumps(log_obj)


def setup_logging(
    level: str = "INFO",
    log_file: Optional[Path] = None,
    use_rich: bool = True,
    daemon_mode: bool = False,
    use_json: bool = False,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
) -> None:
    """
    Set up logging configuration for the application.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file
        use_rich: Whether to use rich formatting if available
        daemon_mode: Whether running in daemon mode (file logging only)
        use_json: Whether to use JSON structured logging for files
        max_bytes: Maximum size of log file before rotation
        backup_count: Number of backup files to keep
    """
    # Convert string level to logging constant
    log_level = getattr(logging, level.upper(), logging.INFO)

    # Root logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Log format
    if daemon_mode:
        # More detailed format for daemon logs
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    else:
        # Simpler format for console
        log_format = "%(levelname)s - %(message)s"

    # Console handler
    if not daemon_mode:
        console_handler: logging.Handler
        if use_rich and RICH_AVAILABLE:
            # Use rich handler for pretty console output
            console_handler = RichHandler(
                console=console,
                show_time=False,
                show_path=False,
                markup=True,
                rich_tracebacks=True,
                tracebacks_show_locals=log_level == logging.DEBUG,
            )
            console_handler.setFormatter(logging.Formatter("%(message)s"))
        else:
            # Use custom ANSI handler
            console_handler = ANSIColorHandler()
            console_handler.setFormatter(logging.Formatter(log_format))

        console_handler.setLevel(log_level)
        root_logger.addHandler(console_handler)

    # File handler with rotation
    if log_file:
        # Create parent directory if it doesn't exist
        log_file.parent.mkdir(parents=True, exist_ok=True)

        # Use rotating file handler
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
        )
        file_handler.setLevel(log_level)

        # Use JSON formatter if requested, otherwise standard format
        if use_json:
            file_handler.setFormatter(JSONFormatter())
        else:
            file_handler.setFormatter(
                logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            )
        root_logger.addHandler(file_handler)

    # Configure specific loggers to reduce noise
    logging.getLogger("websockets").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)

    # Suppress transformers warnings as before
    import os

    os.environ["TRANSFORMERS_VERBOSITY"] = "error"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the given name."""
    return logging.getLogger(name)


# Convenience functions for consistent output formatting
def log_success(message: str, logger: Optional[logging.Logger] = None) -> None:
    """Log a success message with appropriate formatting."""
    if logger:
        logger.info(message)
    else:
        if RICH_AVAILABLE and console:
            console.print(f"[green]✓[/green] [cream]{message}[/cream]")
        else:
            sys.stdout.write(
                f"{ANSI_COLORS['green']}✓{ANSI_COLORS['reset']} {ANSI_COLORS['cream']}{message}{ANSI_COLORS['reset']}\n"
            )
            sys.stdout.flush()


def log_error(message: str, logger: Optional[logging.Logger] = None) -> None:
    """Log an error message with appropriate formatting."""
    if logger:
        logger.error(message)
    else:
        if RICH_AVAILABLE and console:
            console.print(f"[red]✗[/red] [cream]{message}[/cream]")
        else:
            sys.stderr.write(
                f"{ANSI_COLORS['red']}✗{ANSI_COLORS['reset']} {ANSI_COLORS['cream']}{message}{ANSI_COLORS['reset']}\n"
            )
            sys.stderr.flush()


def log_warning(message: str, logger: Optional[logging.Logger] = None) -> None:
    """Log a warning message with appropriate formatting."""
    if logger:
        logger.warning(message)
    else:
        if RICH_AVAILABLE and console:
            console.print(f"[yellow]![/yellow] [cream]{message}[/cream]")
        else:
            sys.stdout.write(
                f"{ANSI_COLORS['amber']}!{ANSI_COLORS['reset']} {ANSI_COLORS['cream']}{message}{ANSI_COLORS['reset']}\n"
            )
            sys.stdout.flush()


def log_info(message: str, logger: Optional[logging.Logger] = None) -> None:
    """Log an info message with appropriate formatting."""
    if logger:
        logger.info(message)
    else:
        if RICH_AVAILABLE and console:
            console.print(f"[cyan]→[/cyan] [cream]{message}[/cream]")
        else:
            sys.stdout.write(
                f"{ANSI_COLORS['cyan']}→{ANSI_COLORS['reset']} {ANSI_COLORS['cream']}{message}{ANSI_COLORS['reset']}\n"
            )
            sys.stdout.flush()


def log_activity(message: str, logger: Optional[logging.Logger] = None) -> None:
    """Log a monitoring/activity message with dim formatting."""
    if logger:
        logger.info(message)
    else:
        if RICH_AVAILABLE and console:
            console.print(f"[dim_cream]{message}[/dim_cream]")
        else:
            sys.stdout.write(f"{ANSI_COLORS['dim_cream']}{message}{ANSI_COLORS['reset']}\n")
            sys.stdout.flush()
