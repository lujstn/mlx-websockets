#!/usr/bin/env python3
"""Command-line interface for MLX WebSockets."""

import argparse
import logging
import sys
from typing import Optional

from mlx_websockets.daemon import (
    find_all_mlx_processes,
    format_runtime,
    format_time,
    get_server_status,
    start_background_server,
    stop_background_server,
)
from mlx_websockets.exceptions import DaemonError, MLXWebSocketError
from mlx_websockets.logging_utils import (
    log_error,
    log_info,
    log_success,
    log_warning,
    setup_logging,
)
from mlx_websockets.server import main as run_server

logger = logging.getLogger(__name__)


def serve_command(args: argparse.Namespace) -> None:
    """Run the WebSocket server with comprehensive process checking."""

    try:
        # Check for existing MLX processes
        all_processes = find_all_mlx_processes()

        # If user specified a port, check if it's in use by MLX
        if args.port and all_processes:
            conflicting = [p for p in all_processes if p.port == args.port]
            if conflicting:
                proc = conflicting[0]
                log_error(
                    f"Cannot start: Port {args.port} is already in use by MLX server\n"
                    f"  PID: {proc.pid}\n"
                    f"  Model: {proc.model or 'unknown'}\n"
                    f"  Type: {'daemon-managed' if proc.is_daemon else 'standalone'}\n\n"
                    f"To stop it: mlx bg stop"
                )
                sys.exit(1)

        # If no port specified but MLX processes exist, let auto-discovery handle it
        if not args.port and all_processes:
            log_info(
                f"Found {len(all_processes)} existing MLX server(s). Auto-discovering available port..."
            )

        # Convert namespace back to args list for server.py
        server_args = []

        if args.model:
            server_args.extend(["--model", args.model])
        if args.host:
            server_args.extend(["--host", args.host])
        if args.port:
            server_args.extend(["--port", str(args.port)])
        else:
            # If no port specified, enable auto-port discovery
            server_args.extend(["--auto-port"])
        if args.trust_remote_code:
            server_args.append("--trust-remote-code")
        if args.tokenizer_config:
            server_args.extend(["--tokenizer-config", args.tokenizer_config])
        if args.chat_template:
            server_args.extend(["--chat-template", args.chat_template])
        if args.max_tokens:
            server_args.extend(["--max-tokens", str(args.max_tokens)])
        if args.temperature:
            server_args.extend(["--temperature", str(args.temperature)])
        if args.seed:
            server_args.extend(["--seed", str(args.seed)])
        if args.debug:
            server_args.append("--debug")

        # Pass args to the original server main function
        sys.argv = ["mlx_websockets.server"] + server_args
        run_server()
    except KeyboardInterrupt:
        log_info("\nServer shutdown requested.")
        sys.exit(0)
    except MLXWebSocketError as e:
        log_error(f"Error: {e}")
        sys.exit(1)
    except OSError as e:
        log_error(f"System error: Unable to start server - {e}")
        logger.debug(f"OS error in serve command: {e}", exc_info=True)
        sys.exit(1)
    except ImportError as e:
        log_error("Missing required dependencies. Please reinstall mlx-websockets.")
        logger.debug(f"Import error: {e}", exc_info=True)
        sys.exit(1)


def background_serve_command(args: argparse.Namespace) -> None:
    """Run the WebSocket server in the background."""
    try:
        server_config = {
            "model": args.model,
            "host": args.host,
            "port": args.port,
            "trust_remote_code": args.trust_remote_code,
            "tokenizer_config": args.tokenizer_config,
            "chat_template": args.chat_template,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "seed": args.seed,
        }

        # Remove None values
        server_config = {k: v for k, v in server_config.items() if v is not None}

        start_background_server(server_config)

        # Get the actual port from the status
        status = get_server_status()
        if status:
            log_success(f"Server started in background on port {status['port']}")
            log_info("To stop: mlx background stop")
        else:
            log_error("Server started but status unavailable")
    except DaemonError as e:
        log_error(str(e))
        sys.exit(1)
    except OSError as e:
        log_error("System error: Unable to start background server")
        logger.debug(f"OS error in background serve: {e}", exc_info=True)
        sys.exit(1)
    except ValueError as e:
        log_error("Invalid configuration provided")
        logger.debug(f"Value error: {e}", exc_info=True)
        sys.exit(1)


def background_stop_command(args: argparse.Namespace) -> None:
    """Stop MLX WebSocket servers."""
    try:
        # Check if we should stop only daemon servers
        daemon_only = getattr(args, "daemon_only", False)

        # First, check what's running
        all_processes = find_all_mlx_processes()
        if not all_processes:
            log_warning("No MLX servers are running")
            return

        # Show what will be stopped
        daemon_count = sum(1 for p in all_processes if p.is_daemon)
        standalone_count = len(all_processes) - daemon_count

        if daemon_only and standalone_count > 0:
            log_info(
                f"Stopping {daemon_count} daemon-managed server(s) (skipping {standalone_count} standalone)"
            )
        else:
            log_info(f"Stopping {len(all_processes)} MLX server(s)...")

        if stop_background_server(daemon_only=daemon_only):
            log_success("MLX server(s) stopped successfully")
        else:
            log_warning("Failed to stop some MLX servers")

    except DaemonError as e:
        log_error(str(e))
        sys.exit(1)
    except OSError as e:
        log_error("System error: Unable to stop server")
        logger.debug(f"OS error stopping server: {e}", exc_info=True)
        sys.exit(1)


def status_command(args: argparse.Namespace) -> None:
    """Check the status of ALL MLX servers (daemon-managed and standalone)."""
    from .process_registry import get_registry

    try:
        # Get all MLX processes
        all_processes = find_all_mlx_processes()

        if not all_processes:
            log_warning("No MLX servers are running")
            return

        # Sort by type (daemon first) and then by port
        all_processes.sort(key=lambda p: (not p.is_daemon, p.port))

        log_success(f"Found {len(all_processes)} MLX server(s) running:")
        log_info("")

        for proc in all_processes:
            if proc.is_daemon:
                log_info("[Daemon-managed]")
                # Get additional info from daemon status
                status = get_server_status()
                if status:
                    log_info(f"  Model: {status.get('model', proc.model or 'unknown')}")
                    log_info(f"  Port: {proc.port}")
                    log_info(f"  PID: {proc.pid}")
                    log_info(f"  Started: {status.get('started', format_time(proc.start_time))}")
                    log_info(f"  Running for: {format_runtime(proc.start_time)}")
                else:
                    # Fallback if daemon status unavailable
                    log_info(f"  Model: {proc.model or 'unknown'}")
                    log_info(f"  Port: {proc.port}")
                    log_info(f"  PID: {proc.pid}")
                    log_info(f"  Started: {format_time(proc.start_time)}")
                    log_info(f"  Running for: {format_runtime(proc.start_time)}")
            else:
                log_info("[Standalone]")
                log_info(f"  Model: {proc.model or 'unknown'}")
                log_info(f"  Port: {proc.port}")
                log_info(f"  PID: {proc.pid}")
                log_info(f"  Started: {format_time(proc.start_time)}")
                log_info(f"  Running for: {format_runtime(proc.start_time)}")
                if proc.cmdline:
                    cmd_preview = " ".join(proc.cmdline[:5])
                    if len(proc.cmdline) > 5:
                        cmd_preview += " ..."
                    log_info(f"  Command: {cmd_preview}")
            log_info("")

        # Check for unregistered processes
        registry = get_registry()
        unregistered = registry.detect_unregistered_mlx_processes()
        if unregistered:
            log_warning(f"\nFound {len(unregistered)} unregistered MLX process(es):")
            for pid, port in unregistered:
                log_warning(f"  PID: {pid}, Port: {port}")
            log_info("\nThese processes are not tracked properly. Consider restarting them.")

    except DaemonError as e:
        log_error(str(e))
        sys.exit(1)
    except OSError as e:
        log_error("System error: Unable to check server status")
        logger.debug(f"OS error checking status: {e}", exc_info=True)
        sys.exit(1)


def cleanup_command(args: argparse.Namespace) -> None:
    """Clean up stale process registry entries."""
    from .process_registry import get_registry

    try:
        registry = get_registry()

        # Get all registered processes
        all_processes = registry.get_all_processes()

        if not all_processes:
            log_info("No process entries to clean up")
            return

        # Find stale entries
        stale_count = 0
        for proc in all_processes:
            if not registry._is_process_running(proc.pid) or args.force:
                if registry.unregister_process(proc.pid):
                    stale_count += 1
                    log_info(f"Cleaned up stale entry for PID {proc.pid}")

        if stale_count > 0:
            log_success(f"Cleaned up {stale_count} stale process entries")
        else:
            log_info("No stale entries found")

        # Also clean up daemon files if stale
        daemon_status = get_server_status()
        if daemon_status and not registry._is_process_running(daemon_status["pid"]):
            from .daemon import get_config_file, get_pid_file

            get_pid_file().unlink(missing_ok=True)
            get_config_file().unlink(missing_ok=True)
            log_info("Cleaned up stale daemon files")

    except Exception as e:
        log_error(f"Failed to clean up: {e}")
        sys.exit(1)


def main(argv: Optional[list[str]] = None) -> None:
    """Main entry point for the CLI."""
    # Setup basic logging for CLI
    setup_logging(level="INFO", use_rich=True)

    parser = argparse.ArgumentParser(
        prog="mlx", description="MLX WebSockets - Streaming server for MLX models"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Run the WebSocket server")
    serve_parser.add_argument(
        "--model",
        type=str,
        default="mlx-community/gemma-3-4b-it-4bit",
        help="The path to the MLX model weights, tokenizer, and config",
    )
    serve_parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind the server to (use 'localhost' for local-only access)",
    )
    serve_parser.add_argument("--port", type=int, default=8765, help="Port to bind the server to")
    serve_parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Enable trusting remote code for tokenizer",
    )
    serve_parser.add_argument(
        "--tokenizer-config",
        type=str,
        help="Tokenizer config.json file",
    )
    serve_parser.add_argument(
        "--chat-template",
        type=str,
        help="Chat template or template name",
    )
    serve_parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum number of tokens to generate",
    )
    serve_parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature",
    )
    serve_parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for generation",
    )
    serve_parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    serve_parser.set_defaults(func=serve_command)

    # Background command with subcommands
    bg_parser = subparsers.add_parser("background", aliases=["bg"], help="Manage background server")
    bg_subparsers = bg_parser.add_subparsers(dest="bg_command", help="Background commands")

    # Background serve
    bg_serve_parser = bg_subparsers.add_parser("serve", help="Run server in background")
    bg_serve_parser.add_argument(
        "--model",
        type=str,
        default="mlx-community/gemma-3-4b-it-4bit",
        help="The path to the MLX model weights, tokenizer, and config",
    )
    bg_serve_parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind the server to (use 'localhost' for local-only access)",
    )
    bg_serve_parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Port to bind the server to (will auto-find if in use)",
    )
    bg_serve_parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Enable trusting remote code for tokenizer",
    )
    bg_serve_parser.add_argument(
        "--tokenizer-config",
        type=str,
        help="Tokenizer config.json file",
    )
    bg_serve_parser.add_argument(
        "--chat-template",
        type=str,
        help="Chat template or template name",
    )
    bg_serve_parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum number of tokens to generate",
    )
    bg_serve_parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature",
    )
    bg_serve_parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for generation",
    )
    bg_serve_parser.set_defaults(func=background_serve_command)

    # Background stop
    bg_stop_parser = bg_subparsers.add_parser(
        "stop", help="Stop ALL MLX servers (both daemon-managed and standalone)"
    )
    bg_stop_parser.add_argument(
        "--daemon-only",
        action="store_true",
        help="Stop only daemon-managed servers, leaving standalone servers running",
    )
    bg_stop_parser.set_defaults(func=background_stop_command)

    # Status command
    status_parser = subparsers.add_parser("status", help="Check server status")
    status_parser.set_defaults(func=status_command)

    # Cleanup command
    cleanup_parser = subparsers.add_parser("cleanup", help="Clean up stale process entries")
    cleanup_parser.add_argument(
        "--force", action="store_true", help="Force cleanup even if processes appear to be running"
    )
    cleanup_parser.set_defaults(func=cleanup_command)

    # Help command
    help_parser = subparsers.add_parser("help", help="Show help information")
    help_parser.set_defaults(func=lambda args: parser.print_help())

    # Parse arguments
    args = parser.parse_args(argv)

    # If no command specified, show help
    if not args.command:
        parser.print_help()
        sys.exit(0)

    # Handle background command without subcommand
    if args.command in ["background", "bg"] and (
        not hasattr(args, "bg_command") or args.bg_command is None
    ):
        bg_parser.print_help()
        sys.exit(0)

    # Execute the command
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
