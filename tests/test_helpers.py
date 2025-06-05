"""Test helper utilities for MLX WebSocket Server tests."""

import asyncio
import sys
import threading
import time
from contextlib import contextmanager
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock


class MockMLXModel:
    """Mock MLX model for testing."""

    def __call__(self, *args, **kwargs):
        return MagicMock()


class MockProcessor:
    """Mock processor for testing."""

    def __init__(self):
        self.tokenizer = MagicMock()
        self.image_processor = MagicMock()

    def __call__(self, *args, **kwargs):
        return {"input_ids": MagicMock()}


class ThreadTracker:
    """Track threads created during tests for cleanup."""

    def __init__(self):
        self.initial_threads = set()
        self.tracked_threads = set()

    def __enter__(self):
        self.initial_threads = set(threading.enumerate())
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Wait for new threads to finish
        current_threads = set(threading.enumerate())
        new_threads = current_threads - self.initial_threads

        for thread in new_threads:
            if thread.is_alive() and not thread.daemon:
                thread.join(timeout=5.0)

    def track(self, thread):
        """Track a specific thread."""
        self.tracked_threads.add(thread)

    def wait_all(self, timeout=5.0):
        """Wait for all tracked threads to complete."""
        for thread in self.tracked_threads:
            if thread.is_alive():
                thread.join(timeout=timeout)


@contextmanager
def mock_mlx_models():
    """Mock MLX model loading and generation for unit tests.

    This mocks mlx_vlm and mlx_lm modules before importing the server,
    which is necessary because the server imports these at module level.
    """
    # Create mock modules
    mock_mlx_vlm = MagicMock()
    mock_mlx_lm = MagicMock()

    # Create mock model and processor
    mock_model = MockMLXModel()
    mock_processor = MockProcessor()

    # Set up load function
    def mock_load(*args, **kwargs):
        return (mock_model, mock_processor)

    mock_mlx_vlm.load = mock_load
    mock_mlx_vlm.generate = MagicMock()

    # Mock mlx_lm (optional text generation)
    mock_mlx_lm.generate = None

    # Insert into sys.modules BEFORE any imports
    sys.modules["mlx_vlm"] = mock_mlx_vlm
    sys.modules["mlx_lm"] = mock_mlx_lm

    # Clear any cached imports
    if "mlx_streaming_server" in sys.modules:
        del sys.modules["mlx_streaming_server"]

    try:
        yield {
            "model": mock_model,
            "processor": mock_processor,
            "generate": mock_mlx_vlm.generate,
            "load": mock_load,
            "text_generate": mock_mlx_lm.generate,
            "mlx_vlm": mock_mlx_vlm,
            "mlx_lm": mock_mlx_lm,
        }
    finally:
        # Clean up sys.modules
        if "mlx_vlm" in sys.modules:
            del sys.modules["mlx_vlm"]
        if "mlx_lm" in sys.modules:
            del sys.modules["mlx_lm"]
        # Don't delete mlx_streaming_server as other tests might need it


def mock_generate_streaming(response_text: str = "Test response", chunk_size: int = 5):
    """Create a mock streaming generator."""

    def generator(*args, **kwargs):
        for i in range(0, len(response_text), chunk_size):
            yield response_text[i : i + chunk_size]

    return generator


def create_simple_websocket(loop: asyncio.AbstractEventLoop, messages: Optional[list[str]] = None):
    """Create a simple mock websocket for testing.

    Args:
        loop: The event loop to use for async operations
        messages: Optional list to collect sent messages

    Returns:
        A mock websocket object with async send method
    """
    if messages is None:
        messages = []

    class SimpleWebSocket:
        def __init__(self):
            self.remote_address = ("127.0.0.1", 12345)
            self.closed = False
            self.close_code = None
            self.close_reason = None

        async def send(self, message: str):
            """Async send that works with asyncio.run_coroutine_threadsafe."""
            if self.closed:
                import websockets

                raise websockets.exceptions.ConnectionClosedOK(self.close_code, self.close_reason)
            messages.append(message)
            # Small delay to simulate network
            await asyncio.sleep(0.001)

        async def close(self, code=1000, reason=""):
            """Close the websocket with proper close frame."""
            self.closed = True
            self.close_code = code
            self.close_reason = reason
            # Simulate close frame sent
            await asyncio.sleep(0.001)

    return SimpleWebSocket()


def run_async_test(coro, timeout=10.0):
    """Run an async test with proper event loop management.

    Args:
        coro: The coroutine to run
        timeout: Maximum time to wait for completion

    Returns:
        The result of the coroutine
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        return loop.run_until_complete(asyncio.wait_for(coro, timeout=timeout))
    finally:
        loop.close()


def setup_event_loop_in_thread():
    """Set up an event loop running in a background thread.

    Returns:
        tuple: (loop, thread) - The event loop and the thread it's running in
    """
    loop = asyncio.new_event_loop()

    def run_loop():
        asyncio.set_event_loop(loop)
        loop.run_forever()

    thread = threading.Thread(target=run_loop, daemon=True)
    thread.start()

    # Give the loop time to start
    time.sleep(0.1)

    return loop, thread


def stop_event_loop(loop, thread, timeout=1.0):
    """Stop an event loop running in a thread.

    Args:
        loop: The event loop to stop
        thread: The thread running the loop
        timeout: Maximum time to wait for thread to stop
    """
    loop.call_soon_threadsafe(loop.stop)
    thread.join(timeout=timeout)


class ServerTestContext:
    """Context manager for server tests with proper setup and teardown."""

    def __init__(self, model_name="test-model"):
        self.model_name = model_name
        self.server = None
        self.loop = None
        self.loop_thread = None
        self.mocks = None

    def __enter__(self):
        # Set up mocks
        self.mock_context = mock_mlx_models()
        self.mocks = self.mock_context.__enter__()

        # Import server after mocking
        from mlx_streaming_server import MLXStreamingServer

        # Create server
        self.server = MLXStreamingServer(model_name=self.model_name, port=0)

        # Set up event loop
        self.loop, self.loop_thread = setup_event_loop_in_thread()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Stop event loop
        if self.loop and self.loop_thread:
            stop_event_loop(self.loop, self.loop_thread)

        # Clean up mocks
        if self.mock_context:
            self.mock_context.__exit__(exc_type, exc_val, exc_tb)

    def create_websocket(self, messages=None):
        """Create a websocket for this test context."""
        return create_simple_websocket(self.loop, messages)

    def process_text(self, websocket, prompt, client_id=None, context=""):
        """Process text through the server."""
        if client_id is None:
            client_id = ("127.0.0.1", 12345)

        stop_event = threading.Event()

        # Initialize client
        with self.server.clients_lock:
            self.server.client_generators[client_id] = []

        # Prepare data
        text_data = {
            "type": "text",
            "timestamp": time.time(),
            "content": prompt,
            "context": context,
            "prompt": prompt,
        }

        # Process in thread
        def run():
            self.server._process_text(text_data, websocket, self.loop, client_id, stop_event)

        thread = threading.Thread(target=run)
        thread.start()
        thread.join(timeout=5.0)

        return stop_event


# Fixtures for different test scenarios
@contextmanager
def benchmark_test_setup():
    """Setup for benchmark tests - minimal mocking, real threading."""
    with ServerTestContext() as ctx:
        yield ctx


@contextmanager
def integration_test_setup():
    """Setup for integration tests - mock ML models but use real threading/async."""
    with ServerTestContext() as ctx:
        yield ctx
