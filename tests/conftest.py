"""
Shared test configuration and fixtures for MLX WebSocket Streaming Server tests
"""

import asyncio
import base64
import io
import warnings
from unittest.mock import Mock

import pytest
from PIL import Image

from .test_helpers import (
    ThreadTracker,
    mock_mlx_base,  # Import the session-scoped base fixture
    mock_mlx_models,  # Import the new pytest fixture
    mock_mlx_models_context,  # Keep for backward compatibility
)

# Suppress warnings from transformers about missing PyTorch/TensorFlow
warnings.filterwarnings("ignore", message="Disabling PyTorch")
warnings.filterwarnings("ignore", message="None of PyTorch, TensorFlow")
warnings.filterwarnings("ignore", message="mx.metal.get_active_memory is deprecated")

# Suppress torchvision beta warnings
try:
    import torchvision

    torchvision.disable_beta_transforms_warning()
except (ImportError, AttributeError):
    # If torchvision isn't available or doesn't have the function, ignore
    pass

# Also suppress via warnings filter as backup
warnings.filterwarnings("ignore", message=".*torchvision.datapoints.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*torchvision.transforms.v2.*", category=UserWarning)


# Note: We use the simpler context manager approach here because
# the mlx_websockets.server module needs mocks in place before
# import time, not just during test execution.
@pytest.fixture
def mock_mlx_models():
    """Mock MLX model loading for all tests.

    This fixture uses a context manager instead of pytest's monkeypatch
    because the server module's lazy loading pattern requires mocks to
    be in sys.modules before the server module is imported.
    """
    with mock_mlx_models_context() as mocks:
        yield mocks["model"], mocks["processor"]


@pytest.fixture
def mock_text_generation():
    """Mock text generation for tests"""
    with mock_mlx_models_context() as mocks:
        # Default generators
        mocks["generate"].return_value = iter(["Test", " ", "response"])
        if hasattr(mocks, "text_generate"):
            mocks["text_generate"].return_value = iter(["Text", " ", "only", " ", "response"])

        yield mocks["generate"], mocks.get("text_generate")


@pytest.fixture
def thread_tracker():
    """Track threads created during tests"""
    with ThreadTracker() as tracker:
        yield tracker


@pytest.fixture
def test_image_base64():
    """Create a test image and return base64 encoded data"""
    img = Image.new("RGB", (100, 100), color="red")
    img_buffer = io.BytesIO()
    img.save(img_buffer, format="PNG")
    img_buffer.seek(0)
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
    return f"data:image/png;base64,{img_base64}"


# Test configuration
class TestConfig:
    """Shared test configuration"""

    DEFAULT_MODEL = "test-model"
    DEFAULT_PORT = 0  # Use random port
    DEFAULT_TEMPERATURE = 0.7
    DEFAULT_MAX_TOKENS = 200

    # Timeout values for tests
    WEBSOCKET_TIMEOUT = 5.0
    INFERENCE_TIMEOUT = 10.0

    # Performance test parameters
    BENCHMARK_ITERATIONS = 10
    LOAD_TEST_CLIENTS = 20

    # Image test sizes
    SMALL_IMAGE = (100, 100)
    MEDIUM_IMAGE = (500, 500)
    LARGE_IMAGE = (1000, 1000)
    XLARGE_IMAGE = (2000, 2000)


# Shared test utilities
def create_test_image(size=(100, 100), color="red", format="PNG"):
    """Helper to create test images"""
    img = Image.new("RGB", size, color=color)
    img_buffer = io.BytesIO()

    if format == "BMP":
        img.save(img_buffer, format=format)
    else:
        img.save(img_buffer, format=format, quality=95)

    img_buffer.seek(0)
    return img, base64.b64encode(img_buffer.getvalue()).decode()


def create_test_messages(message_type="text", count=1, content=None):
    """Helper to create test WebSocket messages"""
    messages = []

    for i in range(count):
        if message_type == "text":
            msg = {"type": "text_input", "content": content or f"Test message {i}", "context": ""}
        elif message_type == "image":
            _, img_base64 = create_test_image()
            msg = {
                "type": "image_input",
                "image": f"data:image/png;base64,{img_base64}",
                "prompt": content or f"Describe image {i}",
            }
        elif message_type == "config":
            msg = {"type": "config", "temperature": 0.8, "maxTokens": 300}
        else:
            msg = {"type": "unknown"}

        messages.append(msg)

    return messages


@pytest.fixture(autouse=True)
async def cleanup_between_tests():
    """Minimal cleanup between tests for better isolation"""
    # Before test
    yield

    # After test - minimal cleanup
    await asyncio.sleep(0.1)  # Brief pause for pending operations

    # Cancel any pending tasks in the current event loop
    try:
        # Get all tasks in current loop and cancel them
        tasks = asyncio.all_tasks(asyncio.get_event_loop())
        for task in tasks:
            if not task.done() and task != asyncio.current_task():
                task.cancel()
    except RuntimeError:
        pass


# Async test helpers
async def simulate_websocket_client(server, messages, timeout=5.0):
    """Simulate a WebSocket client sending messages"""
    responses = []

    class MockWebSocket:
        def __init__(self):
            self.remote_address = ("127.0.0.1", 12345)
            self.closed = False

        async def send(self, msg):
            if not self.closed:
                responses.append(msg)

        async def close(self):
            self.closed = True

        def __aiter__(self):
            return self

        async def __anext__(self):
            if messages and not self.closed:
                return messages.pop(0)
            raise StopAsyncIteration

    mock_ws = MockWebSocket()

    try:
        await asyncio.wait_for(server.handle_client(mock_ws, "/"), timeout=timeout)
    except asyncio.TimeoutError:
        pass

    return responses


# Performance measurement decorators
def measure_time(func):
    """Decorator to measure function execution time"""

    async def async_wrapper(*args, **kwargs):
        start = asyncio.get_event_loop().time()
        result = await func(*args, **kwargs)
        end = asyncio.get_event_loop().time()
        print(f"\n{func.__name__} took {end - start:.3f}s")
        return result

    def sync_wrapper(*args, **kwargs):
        import time

        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"\n{func.__name__} took {end - start:.3f}s")
        return result

    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper
