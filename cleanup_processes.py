#!/usr/bin/env python3
"""
Script to find and clean up hanging Python processes from the tests
"""

import os
import signal
import subprocess
import sys
import time


def find_python_processes():
    """Find all Python processes related to pytest or mlx_streaming_server"""
    try:
        # Get all processes
        result = subprocess.run(["ps", "aux"], capture_output=True, text=True)

        processes = []
        for line in result.stdout.split("\n"):
            # Look for Python processes that might be from our tests
            if any(keyword in line for keyword in ["pytest", "mlx_streaming_server", "test_"]):
                if "python" in line.lower():
                    parts = line.split()
                    if len(parts) > 1:
                        pid = parts[1]
                        processes.append((pid, line))

        return processes
    except Exception as e:
        print(f"Error finding processes: {e}")
        return []


def kill_process(pid):
    """Kill a process by PID"""
    try:
        os.kill(int(pid), signal.SIGTERM)
        time.sleep(0.5)
        # Check if still running and force kill if needed
        try:
            os.kill(int(pid), 0)  # Check if process exists
            print(f"  Process {pid} didn't terminate, sending SIGKILL...")
            os.kill(int(pid), signal.SIGKILL)
        except ProcessLookupError:
            pass  # Process already terminated
        return True
    except ProcessLookupError:
        print(f"  Process {pid} already terminated")
        return True
    except Exception as e:
        print(f"  Error killing process {pid}: {e}")
        return False


def main():
    print("Looking for hanging Python test processes...")
    print("-" * 80)

    processes = find_python_processes()

    if not processes:
        print("No hanging test processes found!")
        return

    print(f"Found {len(processes)} potentially hanging process(es):\n")

    for i, (pid, line) in enumerate(processes):
        # Truncate long lines for display
        display_line = line[:120] + "..." if len(line) > 120 else line
        print(f"{i+1}. PID {pid}: {display_line}")

    print("\nDo you want to terminate these processes?")
    print("Options:")
    print("  a - Kill all processes")
    print("  n - Don't kill any (exit)")
    print("  1,2,3... - Kill specific processes by number")

    choice = input("\nYour choice: ").strip().lower()

    if choice == "n":
        print("Exiting without killing any processes.")
        return
    elif choice == "a":
        to_kill = list(range(len(processes)))
    else:
        try:
            # Parse comma-separated numbers
            to_kill = [int(x.strip()) - 1 for x in choice.split(",")]
        except ValueError:
            print("Invalid input. Exiting.")
            return

    print("\nKilling selected processes...")
    killed = 0
    for idx in to_kill:
        if 0 <= idx < len(processes):
            pid, line = processes[idx]
            print(f"\nKilling PID {pid}...")
            if kill_process(pid):
                killed += 1

    print(f"\nSuccessfully terminated {killed} process(es).")

    # Also clean up any orphaned threads by looking for specific patterns
    print("\nChecking for any remaining WebSocket or MLX processes...")

    # Look for any websocket or mlx-related processes
    result = subprocess.run(
        ["lsof", "-i", ":8765"], capture_output=True, text=True  # Check if port 8765 is in use
    )

    if result.stdout:
        print("Found processes using port 8765:")
        print(result.stdout)
        print("\nYou may want to run this script again to clean these up.")


if __name__ == "__main__":
    # Check if running as root (not recommended but sometimes needed)
    if os.geteuid() == 0:
        print("Warning: Running as root. Be careful!")

    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
