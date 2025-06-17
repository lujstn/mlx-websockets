#!/usr/bin/env python3
"""
Script to find and clean up hanging Python processes from the tests
"""

import argparse
import logging
import os
import signal
import subprocess
import sys
import time

# Try to import from the package, fall back to basic logging
try:
    from mlx_websockets.logging_utils import (
        get_logger,
        log_error,
        log_info,
        log_success,
        log_warning,
        setup_logging,
    )

    LOGGING_AVAILABLE = True
except ImportError:
    LOGGING_AVAILABLE = False
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    get_logger = logging.getLogger
    log_info = log_warning = log_error = log_success = print


def find_python_processes(logger):
    """Find all Python processes related to pytest or mlx_websockets"""
    try:
        # Get all processes
        result = subprocess.run(["ps", "aux"], capture_output=True, text=True)

        processes = []
        for line in result.stdout.split("\n"):
            # Look for Python processes that might be from our tests
            if any(keyword in line for keyword in ["pytest", "test_", "mlx_websockets"]):
                if "python" in line.lower():
                    parts = line.split()
                    if len(parts) > 1:
                        pid = parts[1]
                        processes.append((pid, line))

        return processes
    except Exception as e:
        logger.error(f"Error finding processes: {e}")
        return []


def kill_process(pid, logger):
    """Kill a process by PID"""
    try:
        os.kill(int(pid), signal.SIGTERM)
        time.sleep(0.5)
        # Check if still running and force kill if needed
        try:
            os.kill(int(pid), 0)  # Check if process exists
            logger.warning(f"Process {pid} didn't terminate, sending SIGKILL...")
            os.kill(int(pid), signal.SIGKILL)
        except ProcessLookupError:
            pass  # Process already terminated
        return True
    except ProcessLookupError:
        logger.info(f"Process {pid} already terminated")
        return True
    except PermissionError:
        logger.error(f"Permission denied to kill process {pid}")
        return False
    except Exception as e:
        logger.error(f"Error killing process {pid}: {e}")
        return False


def main(verbose=False):
    """Main function to clean up processes"""
    # Set up logging
    if LOGGING_AVAILABLE:
        setup_logging(debug=verbose)
        logger = get_logger(__name__)
    else:
        logger = get_logger(__name__)
        if verbose:
            logger.setLevel(logging.DEBUG)

    log_info("Looking for hanging Python test processes...")
    if not LOGGING_AVAILABLE:
        print("-" * 80)

    processes = find_python_processes(logger)

    if not processes:
        log_success("No hanging test processes found!")
        return

    log_warning(f"Found {len(processes)} potentially hanging process(es):\n")

    for i, (pid, line) in enumerate(processes):
        # Truncate long lines for display
        display_line = line[:120] + "..." if len(line) > 120 else line
        logger.info(f"{i + 1}. PID {pid}: {display_line}")

    print("\nDo you want to terminate these processes?")
    print("Options:")
    print("  a - Kill all processes")
    print("  n - Don't kill any (exit)")
    print("  1,2,3... - Kill specific processes by number")

    choice = input("\nYour choice: ").strip().lower()

    if choice == "n":
        log_info("Exiting without killing any processes.")
        return
    elif choice == "a":
        to_kill = list(range(len(processes)))
    else:
        try:
            # Parse comma-separated numbers
            to_kill = [int(x.strip()) - 1 for x in choice.split(",")]
        except ValueError:
            log_error("Invalid input. Exiting.")
            return

    log_info("\nKilling selected processes...")
    killed = 0
    for idx in to_kill:
        if 0 <= idx < len(processes):
            pid, line = processes[idx]
            logger.info(f"Killing PID {pid}...")
            if kill_process(pid, logger):
                killed += 1

    log_success(f"Successfully terminated {killed} process(es).")

    # Also clean up any orphaned threads by looking for specific patterns
    log_info("\nChecking for any remaining WebSocket or MLX processes...")

    # Look for any websocket or mlx-related processes
    try:
        result = subprocess.run(
            ["lsof", "-i", ":8765"],
            capture_output=True,
            text=True,  # Check if port 8765 is in use
        )

        if result.stdout:
            log_warning("Found processes using port 8765:")
            logger.info(result.stdout)
            log_info("\nYou may want to run this script again to clean these up.")
    except Exception as e:
        logger.debug(f"Could not check port 8765: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean up hanging Python test processes")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")
    args = parser.parse_args()

    # Check if running as root (not recommended but sometimes needed)
    if os.geteuid() == 0:
        log_warning("Warning: Running as root. Be careful!")

    try:
        main(verbose=args.verbose)
    except KeyboardInterrupt:
        log_info("\nInterrupted by user.")
    except Exception as e:
        log_error(f"Error: {e}")
        sys.exit(1)
