#!/bin/bash
# Force cleanup of any hanging Python test processes

echo "Force cleaning up Python test processes..."

# Kill any pytest processes
echo "Looking for pytest processes..."
pkill -f pytest || echo "No pytest processes found"

# Kill any processes running our test files
echo "Looking for test file processes..."
pkill -f "test_inference.py" || echo "No test_inference.py processes"
pkill -f "test_integration.py" || echo "No test_integration.py processes"
pkill -f "test_benchmarks.py" || echo "No test_benchmarks.py processes"
pkill -f "mlx_streaming_server.py" || echo "No mlx_streaming_server.py processes"

# Check for processes on our WebSocket port
echo "Checking port 8765..."
lsof -ti:8765 | xargs kill -9 2>/dev/null || echo "Port 8765 is clear"

# Clean up any Python processes that might be stuck
echo "Looking for stuck Python processes with our code..."
ps aux | grep -E "(pytest|test_.*\.py|mlx_streaming)" | grep -v grep | awk '{print $2}' | xargs kill -9 2>/dev/null || echo "No stuck processes found"

echo "Cleanup complete!"

# Show any remaining Python processes (for verification)
echo -e "\nRemaining Python processes:"
ps aux | grep python | grep -v grep | head -5 || echo "No Python processes found"