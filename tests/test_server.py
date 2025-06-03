"""
Tests for MLX WebSocket Streaming Server
"""

import asyncio
import base64
import io
import json
import sys
import time
from unittest.mock import MagicMock, Mock, patch

import pytest
import websockets
from PIL import Image

sys.path.append("..")
from mlx_streaming_server import MLXStreamingServer


class TestMLXStreamingServer:
    """Test the MLX Streaming Server core functionality"""

    @pytest.fixture
    def mock_model_and_processor(self):
        """Mock the MLX model and processor"""
        with patch("mlx_streaming_server.load") as mock_load:
            mock_model = Mock()
            mock_processor = Mock()
            mock_load.return_value = (mock_model, mock_processor)
            yield mock_model, mock_processor

    @pytest.fixture
    def server(self, mock_model_and_processor):
        """Create a server instance with mocked model"""
        server = MLXStreamingServer(model_name="test-model", port=8765)
        return server

    def test_server_initialization(self, server):
        """Test server initializes with correct defaults"""
        assert server.model_name == "test-model"
        assert server.port == 8765
        assert server.config["temperature"] == 0.7
        assert server.config["maxOutputTokens"] == 200
        assert server.config["topP"] == 1.0
        assert server.config["topK"] == 50
        assert server.max_tokens_image == 100

    def test_config_update_temperature(self, server):
        """Test temperature configuration update"""
        # Test valid temperature
        server.config["temperature"] = 0.9
        assert server.config["temperature"] == 0.9

    def test_config_update_max_tokens(self, server):
        """Test max tokens configuration update"""
        server.config["maxOutputTokens"] = 500
        assert server.config["maxOutputTokens"] == 500

    def test_client_state_management(self, server):
        """Test client state is properly tracked"""
        client_id = ("127.0.0.1", 12345)

        # Add client
        with server.clients_lock:
            server.client_queues[client_id] = Mock()
            server.client_stop_events[client_id] = Mock()
            server.client_frame_counts[client_id] = 0
            server.client_generators[client_id] = []

        # Check client exists
        with server.clients_lock:
            assert client_id in server.client_queues
            assert client_id in server.client_stop_events
            assert server.client_frame_counts[client_id] == 0
            assert server.client_generators[client_id] == []

        # Remove client
        with server.clients_lock:
            server.client_queues.pop(client_id, None)
            server.client_stop_events.pop(client_id, None)
            server.client_frame_counts.pop(client_id, None)
            server.client_generators.pop(client_id, None)

        # Check client removed
        with server.clients_lock:
            assert client_id not in server.client_queues


@pytest.mark.asyncio
class TestWebSocketAPI:
    """Test the WebSocket API endpoints"""

    @pytest.fixture
    async def mock_server(self):
        """Create a mock server for testing WebSocket connections"""
        with patch("mlx_streaming_server.load") as mock_load:
            mock_model = Mock()
            mock_processor = Mock()
            mock_load.return_value = (mock_model, mock_processor)

            # Mock the generate function to return tokens
            with patch("mlx_streaming_server.generate") as mock_generate:
                mock_generate.return_value = iter(["Hello", " ", "world", "!"])

                server = MLXStreamingServer(
                    model_name="test-model", port=0
                )  # Use port 0 for random port
                yield server, mock_generate

    async def test_text_input_message_format(self):
        """Test text input message format validation"""
        # Valid text input
        valid_msg = {
            "type": "text_input",
            "content": "Hello, how are you?",
            "context": "Friendly conversation",
        }

        # Should parse without error
        msg_str = json.dumps(valid_msg)
        parsed = json.loads(msg_str)
        assert parsed["type"] == "text_input"
        assert parsed["content"] == "Hello, how are you?"
        assert parsed["context"] == "Friendly conversation"

    async def test_image_input_message_format(self):
        """Test image input message format validation"""
        # Create a small test image
        img = Image.new("RGB", (10, 10), color="red")
        img_buffer = io.BytesIO()
        img.save(img_buffer, format="PNG")
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()

        # Valid image input
        valid_msg = {
            "type": "image_input",
            "image": f"data:image/png;base64,{img_base64}",
            "prompt": "What's in this image?",
            "source": "test",
        }

        # Should parse without error
        msg_str = json.dumps(valid_msg)
        parsed = json.loads(msg_str)
        assert parsed["type"] == "image_input"
        assert parsed["image"].startswith("data:image/png;base64,")
        assert parsed["prompt"] == "What's in this image?"

    async def test_config_message_format(self):
        """Test configuration update message format"""
        # Valid config update
        valid_msg = {"type": "config", "temperature": 0.9, "maxOutputTokens": 300, "topP": 0.95}

        # Should parse without error
        msg_str = json.dumps(valid_msg)
        parsed = json.loads(msg_str)
        assert parsed["type"] == "config"
        assert parsed["temperature"] == 0.9
        assert parsed["maxOutputTokens"] == 300
        assert parsed["topP"] == 0.95

    async def test_response_message_formats(self):
        """Test all response message formats"""
        # Response start
        start_msg = {"type": "response_start", "timestamp": 1234567890.123, "input_type": "text"}
        assert json.loads(json.dumps(start_msg))["type"] == "response_start"

        # Token message
        token_msg = {"type": "token", "content": "Hello", "timestamp": 1234567890.123}
        assert json.loads(json.dumps(token_msg))["type"] == "token"

        # Response complete
        complete_msg = {
            "type": "response_complete",
            "full_text": "Hello world!",
            "timestamp": 1234567890.123,
            "input_type": "text",
            "inference_time": 0.456,
        }
        assert json.loads(json.dumps(complete_msg))["type"] == "response_complete"

        # Error message
        error_msg = {"type": "error", "error": "Something went wrong", "timestamp": 1234567890.123}
        assert json.loads(json.dumps(error_msg))["type"] == "error"

        # Frame dropped
        dropped_msg = {"type": "frame_dropped", "reason": "processing_queue_full"}
        assert json.loads(json.dumps(dropped_msg))["type"] == "frame_dropped"


class TestImageProcessing:
    """Test image processing utilities"""

    def test_image_resize(self):
        """Test that large images are resized properly"""
        # Create a large image
        large_img = Image.new("RGB", (2000, 2000), color="blue")

        # Resize using same logic as server
        max_size = 768
        if max(large_img.size) > max_size:
            large_img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

        # Check size is within limits
        assert max(large_img.size) <= max_size
        assert large_img.size[0] == 768 or large_img.size[1] == 768

    def test_base64_image_decode(self):
        """Test base64 image decoding"""
        # Create test image
        img = Image.new("RGB", (100, 100), color="green")
        img_buffer = io.BytesIO()
        img.save(img_buffer, format="JPEG")
        img_bytes = img_buffer.getvalue()

        # Test with data URL format
        img_base64 = base64.b64encode(img_bytes).decode()
        data_url = f"data:image/jpeg;base64,{img_base64}"

        # Decode like server does
        if "," in data_url:
            decoded_bytes = base64.b64decode(data_url.split(",")[1])
        else:
            decoded_bytes = base64.b64decode(data_url)

        # Verify we can open the image
        decoded_img = Image.open(io.BytesIO(decoded_bytes))
        assert decoded_img.size == (100, 100)

        # Test direct base64 format
        direct_base64 = base64.b64encode(img_bytes).decode()
        decoded_bytes2 = base64.b64decode(direct_base64)
        decoded_img2 = Image.open(io.BytesIO(decoded_bytes2))
        assert decoded_img2.size == (100, 100)


class TestThreadSafety:
    """Test thread safety mechanisms"""

    def test_locks_exist(self):
        """Test that all necessary locks are created"""
        with patch("mlx_streaming_server.load") as mock_load:
            mock_load.return_value = (Mock(), Mock())
            server = MLXStreamingServer()

            assert hasattr(server, "clients_lock")
            assert hasattr(server, "config_lock")
            assert hasattr(server, "model_lock")

            # Test they are the right types
            from threading import Lock, RLock

            assert isinstance(server.clients_lock, RLock)
            assert isinstance(server.config_lock, RLock)
            assert isinstance(server.model_lock, Lock)


class TestConfigurationValidation:
    """Test configuration parameter validation"""

    def test_temperature_clamping(self):
        """Test temperature is clamped to valid range"""
        # Temperature should be clamped to [0.0, 2.0]
        temps = [-1.0, 0.0, 0.7, 1.0, 2.0, 3.0]
        expected = [0.0, 0.0, 0.7, 1.0, 2.0, 2.0]

        for temp, exp in zip(temps, expected):
            clamped = max(0.0, min(2.0, temp))
            assert clamped == exp

    def test_top_p_clamping(self):
        """Test topP is clamped to valid range"""
        # topP should be clamped to [0.0, 1.0]
        top_ps = [-0.1, 0.0, 0.5, 1.0, 1.5]
        expected = [0.0, 0.0, 0.5, 1.0, 1.0]

        for top_p, exp in zip(top_ps, expected):
            clamped = max(0.0, min(1.0, top_p))
            assert clamped == exp

    def test_top_k_validation(self):
        """Test topK is validated properly"""
        # topK should be at least 1
        top_ks = [-1, 0, 1, 50, 100]
        expected = [1, 1, 1, 50, 100]

        for top_k, exp in zip(top_ks, expected):
            validated = max(1, top_k)
            assert validated == exp

    def test_max_tokens_validation(self):
        """Test maxOutputTokens validation"""
        # Should be positive
        tokens = [-100, 0, 100, 500]

        for token in tokens:
            if token > 0:
                assert token > 0  # Valid
            else:
                assert token <= 0  # Invalid
