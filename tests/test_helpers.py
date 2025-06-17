"""Test helper utilities for MLX WebSocket Server tests."""

import asyncio
import json
import sys
import threading
import time
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any, Optional
from unittest.mock import MagicMock, Mock

import pytest
import websockets


class MockMLXModel:
    """Test MLX model that generates predictable output."""

    def __init__(self, response_pattern: str = "Test response {}"):
        self.response_pattern = response_pattern
        self.call_count = 0

    def __call__(self, *args, **kwargs):
        self.call_count += 1
        # Return a simple object that mlx models would return
        return type("ModelOutput", (), {"shape": (1, 10)})()


class MockProcessor:
    """Test processor that returns valid data structures."""

    def __init__(self):
        self.tokenizer = type(
            "Tokenizer", (), {"decode": lambda x: f"decoded_{x}", "encode": lambda x: [1, 2, 3]}
        )()
        self.image_processor = type(
            "ImageProcessor", (), {"preprocess": lambda x: {"pixel_values": [[1, 2, 3]]}}
        )()

    def __call__(self, *args, **kwargs):
        # Return valid input structure
        return {"input_ids": [[1, 2, 3]]}


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


def create_test_generator(
    response_text: str = "Test response", chunk_size: int = 5
) -> Generator[str, None, None]:
    """Create a test generator that yields text chunks."""
    for i in range(0, len(response_text), chunk_size):
        yield response_text[i : i + chunk_size]
        # Simulate processing time
        time.sleep(0.01)


# Session-scoped base mocks for expensive setup
@pytest.fixture(scope="session")
def mock_mlx_base():
    """Session-level mock setup for MLX models."""
    # Create test model and processor once per session
    test_model = MockMLXModel()
    test_processor = MockProcessor()

    # Create mock modules
    mock_modules = {
        "mlx_vlm": type("Module", (), {})(),
        "mlx_lm": type("Module", (), {})(),
        "mlx": type("Module", (), {})(),
        "mlx_core": type("Module", (), {})(),
        "mlx_metal": type("Module", (), {})(),
        "pil_image": type("Module", (), {})(),
    }

    # Configure PIL mock
    mock_img_instance = MagicMock()
    mock_img_instance.size = (100, 100)
    mock_img_instance.thumbnail = MagicMock()
    mock_img_instance.save = MagicMock()

    mock_modules["pil_image"].open = MagicMock(return_value=mock_img_instance)
    mock_modules["pil_image"].new = MagicMock(return_value=mock_img_instance)
    mock_modules["pil_image"].Resampling = type("Resampling", (), {"LANCZOS": 1})()

    # Configure mlx mocks
    mock_modules["mlx_core"].metal = mock_modules["mlx_metal"]
    mock_modules["mlx_core"].get_active_memory = lambda: 1024 * 1024 * 1024
    mock_modules["mlx_metal"].get_active_memory = lambda: 1024 * 1024 * 1024
    mock_modules["mlx_metal"].clear_cache = lambda: None
    mock_modules["mlx"].core = mock_modules["mlx_core"]
    mock_modules["mlx"].metal = mock_modules["mlx_metal"]
    mock_modules["mlx"].get_active_memory = lambda: 1024 * 1024 * 1024
    mock_modules["mlx"].clear_cache = lambda: None

    return {"model": test_model, "processor": test_processor, "modules": mock_modules}


@pytest.fixture
def mock_mlx_models(mock_mlx_base, monkeypatch):
    """Per-test fixture for mocking MLX models using pytest's monkeypatch.

    This ensures mocking happens at test execution time, not collection time.
    """
    # Collection safety check
    if hasattr(pytest, "_in_collection") and pytest._in_collection:
        pytest.skip("Cannot mock during collection phase")
        return

    # Extract base mocks
    test_model = mock_mlx_base["model"]
    test_processor = mock_mlx_base["processor"]
    mock_modules = mock_mlx_base["modules"]

    # Define mock functions
    def mock_load(*args, **kwargs):
        return (test_model, test_processor)

    def mock_generate(model, processor, image=None, prompt="", max_tokens=100, **kwargs):
        """Generate function that returns a real generator."""
        response = f"Generated response for: {prompt[:30]}..."
        return create_test_generator(response)

    # Configure module functions
    mock_modules["mlx_vlm"].load = mock_load
    mock_modules["mlx_vlm"].generate = mock_generate
    mock_modules["mlx_lm"].generate = mock_generate

    # Patch sys.modules
    for module_name, module_obj in mock_modules.items():
        if module_name == "pil_image":
            monkeypatch.setitem(sys.modules, "PIL.Image", module_obj)
        else:
            monkeypatch.setitem(sys.modules, module_name.replace("_", "."), module_obj)

    # Ensure server module is reimported if needed
    if "mlx_websockets.server" in sys.modules:
        monkeypatch.delitem(sys.modules, "mlx_websockets.server")
    if "mlx_websockets" in sys.modules:
        monkeypatch.delitem(sys.modules, "mlx_websockets")

    # Import and patch server module - but only if we're in a test
    try:
        from mlx_websockets.server import _import_dependencies

        _import_dependencies(debug=True)

        from mlx_websockets import server

        # Use monkeypatch for clean patching
        monkeypatch.setattr(server, "load", mock_load, raising=False)
        monkeypatch.setattr(server, "generate", mock_generate, raising=False)
        monkeypatch.setattr(server, "mx", mock_modules["mlx"], raising=False)
        monkeypatch.setattr(server, "Image", mock_modules["pil_image"], raising=False)
        monkeypatch.setattr(server, "text_generate", mock_modules["mlx_lm"].generate, raising=False)
        monkeypatch.setattr(server, "_dependencies_loaded", True, raising=False)

    except ImportError:
        # Server not imported yet, which is fine
        pass

    # Store reference for test access
    mock_generate._test_reference = True

    yield {
        "model": test_model,
        "processor": test_processor,
        "generate": mock_generate,
        "load": mock_load,
        "text_generate": mock_modules["mlx_lm"].generate,
        "mlx_vlm": mock_modules["mlx_vlm"],
        "mlx_lm": mock_modules["mlx_lm"],
        "mlx": mock_modules["mlx"],
        "pil_image": mock_modules["pil_image"],
    }

    # Cleanup is handled automatically by monkeypatch


@contextmanager
def mock_mlx_models_context():
    """Legacy context manager for backward compatibility.

    New tests should use the pytest fixture instead.
    """
    # Create test modules with minimal functionality
    mock_mlx_vlm = type("Module", (), {})()
    mock_mlx_lm = type("Module", (), {})()
    mock_mlx = type("Module", (), {})()
    mock_mlx_core = type("Module", (), {})()
    mock_mlx_metal = type("Module", (), {})()

    # Mock PIL Image module
    mock_pil_image = type("Module", (), {})()
    mock_img_instance = MagicMock()
    mock_img_instance.size = (100, 100)
    mock_img_instance.thumbnail = MagicMock()
    mock_img_instance.save = MagicMock()

    mock_pil_image.open = MagicMock(return_value=mock_img_instance)
    mock_pil_image.new = MagicMock(return_value=mock_img_instance)
    mock_pil_image.Resampling = type("Resampling", (), {"LANCZOS": 1})()

    # Store original PIL if it exists
    original_pil = sys.modules.get("PIL.Image")

    # Create test model and processor
    test_model = MockMLXModel()
    test_processor = MockProcessor()

    # Set up load function that returns our test implementations
    def mock_load(*args, **kwargs):
        return (test_model, test_processor)

    # Set up generate function that returns a real generator
    def mock_generate(model, processor, image=None, prompt="", max_tokens=100, **kwargs):
        """Generate function that returns a real generator."""
        response = f"Generated response for: {prompt[:30]}..."
        return create_test_generator(response)

    mock_mlx_vlm.load = mock_load
    mock_mlx_vlm.generate = mock_generate

    # Mock mlx_lm (optional text generation)
    mock_mlx_lm.generate = mock_generate  # Use same generator

    # Mock mlx.core
    mock_mlx_core.metal = mock_mlx_metal
    mock_mlx_core.get_active_memory = lambda: 1024 * 1024 * 1024  # 1GB for mx.get_active_memory()
    mock_mlx_metal.get_active_memory = lambda: 1024 * 1024 * 1024  # 1GB
    mock_mlx_metal.clear_cache = lambda: None
    mock_mlx.core = mock_mlx_core
    mock_mlx.metal = mock_mlx_metal  # Also add metal directly to mlx
    mock_mlx.get_active_memory = lambda: 1024 * 1024 * 1024
    mock_mlx.clear_cache = lambda: None

    # Insert into sys.modules BEFORE any imports
    sys.modules["mlx_vlm"] = mock_mlx_vlm
    sys.modules["mlx_lm"] = mock_mlx_lm
    sys.modules["mlx"] = mock_mlx
    sys.modules["mlx.core"] = mock_mlx_core
    sys.modules["mlx.core.metal"] = mock_mlx_metal
    sys.modules["PIL.Image"] = mock_pil_image

    # Clear any cached imports
    if "mlx_websockets.server" in sys.modules:
        del sys.modules["mlx_websockets.server"]
    if "mlx_websockets" in sys.modules:
        del sys.modules["mlx_websockets"]

    # Store a reference for test access
    mock_mlx_models_context._mock_generate = mock_generate

    try:
        # Import server module and initialize dependencies
        # This ensures the server's global variables are properly set
        from mlx_websockets.server import _import_dependencies

        _import_dependencies(debug=True)

        # Now patch the server module's globals with our mocks
        import mlx_websockets.server as server_module

        server_module.load = mock_load
        server_module.generate = mock_generate
        server_module.mx = mock_mlx
        server_module.Image = mock_pil_image
        server_module.text_generate = mock_mlx_lm.generate
        server_module._dependencies_loaded = True

        yield {
            "model": test_model,
            "processor": test_processor,
            "generate": mock_generate,
            "load": mock_load,
            "text_generate": mock_mlx_lm.generate,
            "mlx_vlm": mock_mlx_vlm,
            "mlx_lm": mock_mlx_lm,
            "mlx": mock_mlx,
            "pil_image": mock_pil_image,
        }
    finally:
        # Clean up sys.modules
        for module in ["mlx_vlm", "mlx_lm", "mlx", "mlx.core", "mlx.core.metal", "PIL.Image"]:
            if module in sys.modules:
                del sys.modules[module]

        # Restore original PIL if it existed
        if original_pil:
            sys.modules["PIL.Image"] = original_pil


class MockFuture:
    """Mock future that accepts timeout parameter like concurrent.futures.Future."""

    def __init__(self, result_value=None, exception=None):
        self._result_value = result_value
        self._exception = exception

    def result(self, timeout=None):
        """Return the result or raise the exception."""
        if self._exception:
            raise self._exception
        return self._result_value

    def set_result(self, value):
        """Set the result value."""
        self._result_value = value

    def set_exception(self, exception):
        """Set the exception to raise."""
        self._exception = exception


class RealWebSocketClient:
    """Real WebSocket client for integration testing."""

    def __init__(self, uri: str):
        self.uri = uri
        self.websocket = None
        self.messages = []
        self.connected = False

    async def connect(self):
        """Connect to the WebSocket server."""
        self.websocket = await websockets.connect(
            self.uri,
            max_size=20 * 1024 * 1024,  # 20MB max message size to match server
        )
        self.connected = True

    async def send(self, message: dict):
        """Send a message to the server."""
        if not self.connected:
            raise RuntimeError("Not connected")
        await self.websocket.send(json.dumps(message))

    async def receive(self, timeout: float = 5.0):
        """Receive a message from the server."""
        if not self.connected:
            raise RuntimeError("Not connected")
        try:
            message = await asyncio.wait_for(self.websocket.recv(), timeout=timeout)
            parsed = json.loads(message)
            self.messages.append(parsed)
            return parsed
        except asyncio.TimeoutError:
            return None

    async def receive_all(self, timeout: float = 10.0):
        """Receive all messages until completion or timeout."""
        messages = []
        end_time = time.time() + timeout

        while time.time() < end_time:
            try:
                msg = await self.receive(timeout=0.5)
                if msg:
                    messages.append(msg)
                    if msg.get("type") == "response_complete":
                        break
            except Exception as e:
                # Only break on connection errors, not timeouts
                if isinstance(e, (websockets.exceptions.ConnectionClosed, ConnectionError)):
                    break
                # Continue on other exceptions like timeouts

        return messages

    async def close(self):
        """Close the connection."""
        if self.websocket:
            await self.websocket.close()
            self.connected = False


async def start_test_server(port: int = 0, model_name: str = "test-model", **server_kwargs):
    """Start a real WebSocket server for testing.

    Returns:
        tuple: (server, actual_port) - The server instance and the port it's running on
    """
    from mlx_websockets.server import MLXStreamingServer

    server = MLXStreamingServer(model_name=model_name, port=port, **server_kwargs)

    # Create a wrapper that handles both old and new websockets API
    async def handler(websocket, path=None):
        # New websockets library doesn't pass path
        if path is None:
            path = "/"
        await server.handle_client(websocket, path)

    # Start server and get actual port
    ws_server = await websockets.serve(
        handler,
        "localhost",
        port,
        max_size=20 * 1024 * 1024,  # 20MB max message size to match server
    )

    # Get the actual port if port was 0
    actual_port = ws_server.sockets[0].getsockname()[1]

    return server, ws_server, actual_port


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


class RealServerTestContext:
    """Context manager for server tests using real WebSocket connections."""

    def __init__(self, model_name="test-model", port=0, **server_kwargs):
        self.model_name = model_name
        self.port = port
        self.server_kwargs = server_kwargs
        self.server = None
        self.ws_server = None
        self.actual_port = None
        self.mocks = None
        self.loop = None

    async def __aenter__(self):
        # Set up mocks for ML models only
        self.mock_context = mock_mlx_models_context()
        self.mocks = self.mock_context.__enter__()

        # Import server after mocking
        from mlx_websockets.server import MLXStreamingServer, _import_dependencies

        # Initialize dependencies
        _import_dependencies(debug=True)

        # Create and start real server
        self.server, self.ws_server, self.actual_port = await start_test_server(
            port=self.port, model_name=self.model_name, **self.server_kwargs
        )

        # Track initial clients for later filtering
        self.initial_clients = set(self.server.client_queues.keys())

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # Shutdown server properly
        if self.server:
            await self.server.shutdown()

        # Close WebSocket server
        if self.ws_server:
            self.ws_server.close()
            await self.ws_server.wait_closed()

        # Clean up mocks
        if self.mock_context:
            self.mock_context.__exit__(exc_type, exc_val, exc_tb)

    def create_client(self, uri=None):
        """Create a real WebSocket client."""
        if uri is None:
            uri = f"ws://localhost:{self.actual_port}"
        return RealWebSocketClient(uri)

    async def test_full_message_flow(self, prompt: str, expected_in_response: str = None):
        """Test a complete message flow with a real client."""
        client = self.create_client()
        await client.connect()

        try:
            # Send message
            await client.send({"type": "text_input", "content": prompt})

            # Receive all responses
            messages = await client.receive_all()

            # Verify we got responses
            assert len(messages) > 0, "No messages received"

            # Check for expected content if provided
            if expected_in_response:
                full_response = "".join(
                    msg.get("content", "") for msg in messages if msg.get("type") == "token"
                )
                assert expected_in_response in full_response, (
                    f"Expected '{expected_in_response}' in response, got: {full_response}"
                )

            return messages

        finally:
            await client.close()


# Helper functions for concurrent testing
async def simulate_concurrent_clients(
    server_port: int, num_clients: int = 5, messages_per_client: int = 3
):
    """Simulate multiple concurrent clients sending messages."""
    clients = []
    tasks = []

    # Create and connect clients
    for _i in range(num_clients):
        client = RealWebSocketClient(f"ws://localhost:{server_port}")
        clients.append(client)

    # Connect all clients
    for client in clients:
        await client.connect()

    # Send messages concurrently
    async def send_messages(client, client_id):
        for msg_id in range(messages_per_client):
            await client.send(
                {"type": "text_input", "content": f"Message {msg_id} from client {client_id}"}
            )
            # Small delay between messages
            await asyncio.sleep(0.1)

    # Start all clients sending
    for i, client in enumerate(clients):
        task = asyncio.create_task(send_messages(client, i))
        tasks.append(task)

    # Wait for all to complete
    await asyncio.gather(*tasks)

    # Collect responses
    all_responses = []
    for client in clients:
        responses = await client.receive_all(timeout=5.0)
        all_responses.append(responses)

    # Close all clients
    for client in clients:
        await client.close()

    return all_responses


def wait_for_thread_count(expected_count: int, timeout: float = 5.0, exclude_main: bool = True):
    """Wait for the thread count to reach expected value."""
    start_time = time.time()

    while time.time() - start_time < timeout:
        current_threads = threading.enumerate()
        if exclude_main:
            current_threads = [t for t in current_threads if t != threading.main_thread()]

        if len(current_threads) <= expected_count:
            return True

        time.sleep(0.1)

    return False
