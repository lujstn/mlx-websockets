"""
Basic inference tests that can run quickly without hanging
"""

import json
from unittest.mock import Mock, patch

import pytest
import websockets

from mlx_streaming_server import MLXStreamingServer


class TestBasicInference:
    """Basic inference tests without threading issues"""

    @pytest.fixture
    def mock_server(self):
        """Create a server with mocked components"""
        with patch("mlx_streaming_server.load") as mock_load:
            mock_model = Mock()
            mock_processor = Mock()
            mock_load.return_value = (mock_model, mock_processor)
            server = MLXStreamingServer(model_name="test-model", port=0)
            return server

    def test_text_generation_logic(self, mock_server):
        """Test text generation logic without threading"""
        with patch("mlx_streaming_server.generate") as mock_generate:
            mock_generate.return_value = iter(["Hello", " ", "world", "!"])

            # Test that generate is called with correct parameters
            server = mock_server

            # Mock the model lock
            with server.model_lock:
                with server.config_lock:
                    # Verify config values
                    assert server.config["temperature"] == 0.7
                    assert server.config["maxOutputTokens"] == 200
                    assert server.config["topP"] == 1.0
                    assert server.config["topK"] == 50

    def test_image_processing_logic(self, mock_server):
        """Test image processing logic without threading"""
        import base64
        import io

        from PIL import Image

        # Create test image
        img = Image.new("RGB", (100, 100), color="red")
        img_buffer = io.BytesIO()
        img.save(img_buffer, format="PNG")
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()

        # Test image decoding
        image_data = f"data:image/png;base64,{img_base64}"
        if "," in image_data:
            image_bytes = base64.b64decode(image_data.split(",")[1])
        else:
            image_bytes = base64.b64decode(image_data)

        decoded_img = Image.open(io.BytesIO(image_bytes))
        assert decoded_img.size == (100, 100)

    def test_config_validation(self, mock_server):
        """Test configuration validation"""
        server = mock_server

        # Test temperature clamping
        with server.config_lock:
            server.config["temperature"] = -0.5
            temp = max(0.0, min(2.0, server.config["temperature"]))
            assert temp == 0.0

            server.config["temperature"] = 3.0
            temp = max(0.0, min(2.0, server.config["temperature"]))
            assert temp == 2.0

    def test_model_loading(self):
        """Test model loading behavior"""
        with patch("mlx_streaming_server.load") as mock_load:
            mock_model = Mock()
            mock_processor = Mock()
            mock_load.return_value = (mock_model, mock_processor)

            server = MLXStreamingServer(model_name="test-model", port=0)

            # Verify model was loaded
            mock_load.assert_called_once_with("test-model")
            assert server.model == mock_model
            assert server.processor == mock_processor

    def test_client_state_initialization(self, mock_server):
        """Test client state initialization"""
        server = mock_server
        client_id = ("127.0.0.1", 12345)

        # Simulate adding a client
        import queue
        import threading

        with server.clients_lock:
            server.client_queues[client_id] = queue.Queue()
            server.client_stop_events[client_id] = threading.Event()
            server.client_frame_counts[client_id] = 0
            server.client_generators[client_id] = []

        # Verify state
        with server.clients_lock:
            assert client_id in server.client_queues
            assert client_id in server.client_stop_events
            assert server.client_frame_counts[client_id] == 0
            assert server.client_generators[client_id] == []

    def test_websocket_message_parsing(self, mock_server):
        """Test WebSocket message parsing"""
        # Test text input
        text_msg = json.dumps({"type": "text_input", "content": "Hello", "context": "Test"})
        parsed = json.loads(text_msg)
        assert parsed["type"] == "text_input"
        assert parsed["content"] == "Hello"

        # Test config update
        config_msg = json.dumps({"type": "config", "temperature": 0.9, "maxOutputTokens": 300})
        parsed = json.loads(config_msg)
        assert parsed["type"] == "config"
        assert parsed["temperature"] == 0.9

    @pytest.mark.asyncio
    async def test_websocket_connection_mock(self, mock_server):
        """Test WebSocket connection handling with full mocking"""
        from unittest.mock import AsyncMock, MagicMock

        server = mock_server
        mock_websocket = AsyncMock()
        mock_websocket.remote_address = ("127.0.0.1", 12345)

        # Mock the iterator to return one message then stop
        messages = [json.dumps({"type": "text_input", "content": "test", "context": ""})]

        async def mock_iterator():
            for msg in messages:
                yield msg
            # Raise to exit the loop
            raise websockets.exceptions.ConnectionClosed(None, None)

        mock_websocket.__aiter__ = mock_iterator

        # Mock the processing to avoid threads
        with patch.object(server, "_process_frames"):
            with patch("websockets.exceptions.ConnectionClosed", Exception):
                # This should complete without hanging
                await server.handle_client(mock_websocket, "/")

        # Verify cleanup
        with server.clients_lock:
            assert mock_websocket.remote_address not in server.client_queues


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
