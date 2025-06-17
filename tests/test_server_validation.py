"""Input validation and configuration tests for MLX WebSocket server."""

import asyncio
import json
import queue
import sys
import time
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from .test_helpers import mock_mlx_models, mock_mlx_models_context


def create_server(**kwargs):
    """Helper to create server with proper mocking."""
    from mlx_websockets.server import MLXStreamingServer, _import_dependencies

    _import_dependencies(debug=True)
    return MLXStreamingServer(**kwargs)


class TestConfigurationValidation:
    """Test configuration validation and updates."""

    @pytest.mark.asyncio
    async def test_candidate_count_validation(self):
        """Test candidateCount configuration validation."""
        with mock_mlx_models_context():
            server = create_server(debug=True)

            websocket = AsyncMock()

            # Test valid candidateCount
            config_update = {"candidateCount": 1}
            await server._handle_config(config_update, websocket)

            assert server.config["candidateCount"] == 1

            # Verify response
            sent_msg = json.loads(websocket.send.call_args[0][0])
            assert sent_msg["type"] == "config_updated"
            assert sent_msg["updated_fields"]["candidateCount"] == 1

            # Test conversion for unsupported value
            websocket.reset_mock()
            config_update = {"candidateCount": 5}
            await server._handle_config(config_update, websocket)

            # Should set to 1 and include conversion message
            assert server.config["candidateCount"] == 1
            sent_msg = json.loads(websocket.send.call_args[0][0])
            assert sent_msg["type"] == "config_updated"
            assert sent_msg["updated_fields"]["candidateCount"] == 1
            assert "conversions" in sent_msg
            assert "candidateCount > 1 not supported, using 1" in sent_msg["conversions"]

    @pytest.mark.asyncio
    async def test_max_tokens_validation(self):
        """Test maxTokens configuration validation."""
        with mock_mlx_models_context():
            server = create_server(debug=True)

            websocket = AsyncMock()

            # Test valid value
            config_update = {"maxTokens": 500}
            await server._handle_config(config_update, websocket)

            assert server.config["maxTokens"] == 500

            # Test invalid negative value
            websocket.reset_mock()
            config_update = {"maxTokens": -100}
            await server._handle_config(config_update, websocket)

            # Should not update (negative values are silently ignored)
            assert server.config["maxTokens"] == 500

            # Test invalid type
            websocket.reset_mock()
            config_update = {"maxTokens": "not_a_number"}
            with patch("mlx_websockets.server.logger") as mock_logger:
                await server._handle_config(config_update, websocket)

                # Should not update
                assert server.config["maxTokens"] == 500

                # Should log warning
                mock_logger.warning.assert_called()

    @pytest.mark.asyncio
    async def test_temperature_validation(self):
        """Test temperature configuration validation."""
        with mock_mlx_models_context():
            server = create_server(debug=True)

            websocket = AsyncMock()

            # Test valid value within range
            config_update = {"temperature": 0.8}
            await server._handle_config(config_update, websocket)

            assert server.config["temperature"] == 0.8

            # Test clamping to max
            websocket.reset_mock()
            config_update = {"temperature": 3.0}
            await server._handle_config(config_update, websocket)

            assert server.config["temperature"] == 1.0  # Clamped to max

            # Test clamping to min
            websocket.reset_mock()
            config_update = {"temperature": -0.5}
            await server._handle_config(config_update, websocket)

            assert server.config["temperature"] == 0.0  # Clamped to min

            # Test invalid type
            websocket.reset_mock()
            config_update = {"temperature": "hot"}
            with patch("mlx_websockets.server.logger") as mock_logger:
                await server._handle_config(config_update, websocket)

                # Should not update
                assert server.config["temperature"] == 0.0

                # Should log warning
                mock_logger.warning.assert_called()

    @pytest.mark.asyncio
    async def test_top_p_validation(self):
        """Test topP configuration validation."""
        with mock_mlx_models_context():
            server = create_server()

            websocket = AsyncMock()

            # Test valid value
            config_update = {"topP": 0.8}
            await server._handle_config(config_update, websocket)

            assert server.config["topP"] == 0.8

            # Test clamping
            websocket.reset_mock()
            config_update = {"topP": 1.5}
            await server._handle_config(config_update, websocket)

            assert server.config["topP"] == 1.0  # Clamped to max

            websocket.reset_mock()
            config_update = {"topP": -0.1}
            await server._handle_config(config_update, websocket)

            assert server.config["topP"] == 0.0  # Clamped to min

    @pytest.mark.asyncio
    async def test_top_k_validation(self):
        """Test topK configuration validation."""
        with mock_mlx_models_context():
            server = create_server()

            websocket = AsyncMock()

            # Test valid value
            config_update = {"topK": 100}
            await server._handle_config(config_update, websocket)

            assert server.config["topK"] == 100

            # Test minimum clamping
            websocket.reset_mock()
            config_update = {"topK": 0}
            await server._handle_config(config_update, websocket)

            assert server.config["topK"] == 1  # Clamped to min

            # Test invalid type
            websocket.reset_mock()
            config_update = {"topK": 10.5}  # Float instead of int
            await server._handle_config(config_update, websocket)

            assert server.config["topK"] == 10  # Converted to int

    @pytest.mark.asyncio
    async def test_penalty_validation(self):
        """Test presence and frequency penalty validation."""
        with mock_mlx_models_context():
            server = create_server(debug=True)

            websocket = AsyncMock()

            # Test repetition penalty
            config_update = {"repetitionPenalty": 1.5}
            await server._handle_config(config_update, websocket)

            assert server.config["repetitionPenalty"] == 1.5

            # Test invalid repetition penalty
            websocket.reset_mock()
            config_update = {"repetitionPenalty": "high"}
            with patch("mlx_websockets.server.logger") as mock_logger:
                await server._handle_config(config_update, websocket)

                # Should not update (keep previous value)
                assert server.config["repetitionPenalty"] == 1.5

                # Should log warning
                mock_logger.warning.assert_called()

    @pytest.mark.asyncio
    async def test_response_modalities_validation(self):
        """Test responseModalities configuration validation."""
        with mock_mlx_models_context():
            server = create_server()

            websocket = AsyncMock()

            # Test valid modalities
            config_update = {"responseModalities": ["TEXT"]}
            await server._handle_config(config_update, websocket)

            assert server.config["responseModalities"] == ["TEXT"]
            # Should not have conversions for valid TEXT-only request
            sent_msg = json.loads(websocket.send.call_args[0][0])
            assert "conversions" not in sent_msg

            # Test multiple modalities with conversion
            websocket.reset_mock()
            config_update = {"responseModalities": ["TEXT", "IMAGE", "AUDIO"]}
            await server._handle_config(config_update, websocket)

            # Should filter to TEXT only
            assert server.config["responseModalities"] == ["TEXT"]
            # Check conversion message sent to websocket
            sent_msg = json.loads(websocket.send.call_args[0][0])
            assert "conversions" in sent_msg
            assert isinstance(sent_msg["conversions"], list)
            assert "Only TEXT modality is currently supported" in sent_msg["conversions"][0]

            # Test invalid modality filtering
            websocket.reset_mock()
            config_update = {"responseModalities": ["TEXT", "INVALID", "VIDEO"]}
            await server._handle_config(config_update, websocket)

            # Should filter out invalid
            assert server.config["responseModalities"] == ["TEXT"]
            sent_msg = json.loads(websocket.send.call_args[0][0])
            # No conversions needed since only TEXT remains after filtering
            assert "conversions" not in sent_msg

            # Test invalid type - responseModalities must be a list
            websocket.reset_mock()
            config_update = {"responseModalities": "TEXT"}  # String instead of list
            await server._handle_config(config_update, websocket)

            # Config should not be updated for invalid type
            assert server.config["responseModalities"] == ["TEXT"]  # Should remain unchanged

    @pytest.mark.asyncio
    async def test_max_tokens_image_validation(self):
        """Test max_tokens_image special configuration."""
        with mock_mlx_models_context():
            server = create_server(debug=True)

            websocket = AsyncMock()

            # Test valid value
            config_update = {"max_tokens_image": 50}
            await server._handle_config(config_update, websocket)

            assert server.max_tokens_image == 50

            # Test invalid negative value
            websocket.reset_mock()
            config_update = {"max_tokens_image": -10}
            with patch("mlx_websockets.server.logger") as mock_logger:
                await server._handle_config(config_update, websocket)

                # Should not update
                assert server.max_tokens_image == 50

                # Should log warning
                mock_logger.warning.assert_called()

    @pytest.mark.asyncio
    async def test_config_update_response(self):
        """Test configuration update response format."""
        with mock_mlx_models_context():
            server = create_server()

            websocket = AsyncMock()

            # Update multiple fields
            config_update = {"temperature": 0.9, "maxTokens": 300, "topP": 0.95}

            await server._handle_config(config_update, websocket)

            # Verify response
            sent_msg = json.loads(websocket.send.call_args[0][0])
            assert sent_msg["type"] == "config_updated"
            assert sent_msg["updated_fields"]["temperature"] == 0.9
            assert sent_msg["updated_fields"]["maxTokens"] == 300
            assert sent_msg["updated_fields"]["topP"] == 0.95
            assert "current_config" in sent_msg
            assert sent_msg["current_config"]["temperature"] == 0.9


class TestMessageValidation:
    """Test WebSocket message validation."""

    @pytest.mark.asyncio
    async def test_frame_dropped_when_queue_full(self):
        """Test frame dropping when processing queue is full."""
        # Create a mock queue that always reports as full
        mock_queue_instance = MagicMock()
        mock_queue_instance.full.return_value = True
        mock_queue_instance.put.side_effect = queue.Full
        mock_queue_instance.get.side_effect = queue.Empty
        mock_queue_instance.empty.return_value = True
        mock_queue_instance.qsize.return_value = 1000  # Over the default limit

        # Remove mlx_websockets modules if they're already imported
        modules_to_remove = [k for k in sys.modules.keys() if k.startswith("mlx_websockets")]
        for module in modules_to_remove:
            del sys.modules[module]

        # Patch Queue at the module level before importing
        with patch("queue.Queue", return_value=mock_queue_instance):
            with mock_mlx_models_context():
                # Now import the server module which will use our mocked Queue
                from mlx_websockets.server import MLXStreamingServer, _import_dependencies

                _import_dependencies(debug=True)

                server = MLXStreamingServer(debug=True)

                websocket = AsyncMock()
                websocket.remote_address = ("127.0.0.1", 12345)

                sent_messages = []

                async def capture_send(msg):
                    sent_messages.append(json.loads(msg))

                websocket.send.side_effect = capture_send
                websocket.close = AsyncMock()

                # Mock the message iteration
                messages = [
                    json.dumps({"type": "image_input", "image": "base64data", "prompt": "Describe"})
                ]

                # Mock the async iteration properly
                websocket.__aiter__ = lambda self: async_messages()

                # Create async iterator
                async def async_messages():
                    for msg in messages:
                        yield msg

                # Run with timeout to prevent hanging
                try:
                    await asyncio.wait_for(server.handle_client(websocket, "/"), timeout=2.0)
                except asyncio.TimeoutError:
                    pass
                except Exception:
                    pass  # Ignore other exceptions as we're testing queue behavior

                # Should have sent frame_dropped message
                dropped_msgs = [msg for msg in sent_messages if msg.get("type") == "frame_dropped"]
                assert (
                    len(dropped_msgs) == 1
                ), f"Expected 1 frame_dropped message, got {len(dropped_msgs)}. All messages: {sent_messages}"
                assert dropped_msgs[0]["reason"] == "processing_queue_full"

    @pytest.mark.asyncio
    async def test_text_queue_full_error(self):
        """Test error when text processing queue is full."""
        # Create a mock queue that always reports as full
        mock_queue_instance = MagicMock()
        mock_queue_instance.full.return_value = True
        mock_queue_instance.put.side_effect = queue.Full
        mock_queue_instance.get.side_effect = queue.Empty
        mock_queue_instance.empty.return_value = True
        mock_queue_instance.qsize.return_value = 1000  # Over the default limit

        # Remove mlx_websockets modules if they're already imported
        modules_to_remove = [k for k in sys.modules.keys() if k.startswith("mlx_websockets")]
        for module in modules_to_remove:
            del sys.modules[module]

        # Patch Queue at the module level before importing
        with patch("queue.Queue", return_value=mock_queue_instance):
            with mock_mlx_models_context():
                # Now import the server module which will use our mocked Queue
                from mlx_websockets.server import MLXStreamingServer, _import_dependencies

                _import_dependencies(debug=True)

                server = MLXStreamingServer(debug=True)

                websocket = AsyncMock()
                websocket.remote_address = ("127.0.0.1", 12345)

                sent_messages = []

                async def capture_send(msg):
                    sent_messages.append(json.loads(msg))

                websocket.send.side_effect = capture_send
                websocket.close = AsyncMock()

                # Mock the message iteration
                messages = [json.dumps({"type": "text_input", "content": "Hello"})]

                # Mock the async iteration properly
                websocket.__aiter__ = lambda self: async_messages()

                # Create async iterator
                async def async_messages():
                    for msg in messages:
                        yield msg

                # Run with timeout to prevent hanging
                try:
                    await asyncio.wait_for(server.handle_client(websocket, "/"), timeout=2.0)
                except asyncio.TimeoutError:
                    pass
                except Exception:
                    pass  # Ignore other exceptions as we're testing queue behavior

                # Should have sent error message
                error_msgs = [msg for msg in sent_messages if msg.get("type") == "error"]
                assert len(error_msgs) == 1
                assert "Processing queue full, please retry" in error_msgs[0]["error"]


class TestInputTypeValidation:
    """Test validation of different input types."""

    @pytest.mark.asyncio
    async def test_screen_frame_validation(self):
        """Test screen_frame message validation."""
        with mock_mlx_models_context():
            server = create_server()

            websocket = AsyncMock()
            websocket.remote_address = ("127.0.0.1", 12345)

            # Valid screen_frame
            valid_message = json.dumps(
                {
                    "type": "screen_frame",
                    "frame": "base64imagedata",
                    "prompt": "What's on screen?",
                    "source": "screen",
                }
            )

            # Missing frame
            invalid_message = json.dumps({"type": "screen_frame", "prompt": "What's on screen?"})

            websocket.__aiter__.return_value = [valid_message, invalid_message]

            sent_messages = []

            async def capture_send(msg):
                sent_messages.append(json.loads(msg))

            websocket.send = capture_send

            # Set up queue
            with server.clients_lock:
                server.client_queues[websocket.remote_address] = Mock(full=Mock(return_value=False))
                server.client_stop_events[websocket.remote_address] = Mock()
                server.client_frame_counts[websocket.remote_address] = 0
                server.client_generators[websocket.remote_address] = []

            await server.handle_client(websocket, "/")

            # Should have error for invalid message
            error_msgs = [msg for msg in sent_messages if msg.get("type") == "error"]
            assert any("Missing image/frame content" in msg["error"] for msg in error_msgs)

    @pytest.mark.asyncio
    async def test_video_frame_validation(self):
        """Test video_frame message validation."""
        # Set up queue to capture data
        captured_data = []

        # Create mock queue instance
        mock_queue_instance = MagicMock()
        mock_queue_instance.full.return_value = False
        mock_queue_instance.put.side_effect = lambda x: captured_data.append(x)
        mock_queue_instance.get.side_effect = queue.Empty
        mock_queue_instance.empty.return_value = True
        mock_queue_instance.qsize.return_value = 0

        # Remove mlx_websockets modules if they're already imported
        modules_to_remove = [k for k in sys.modules.keys() if k.startswith("mlx_websockets")]
        for module in modules_to_remove:
            del sys.modules[module]

        # Patch Queue before importing server
        with patch("queue.Queue", return_value=mock_queue_instance):
            with mock_mlx_models_context():
                # Import server after patching Queue
                from mlx_websockets.server import MLXStreamingServer, _import_dependencies

                _import_dependencies(debug=True)

                server = MLXStreamingServer(debug=True)

                websocket = AsyncMock()
                websocket.remote_address = ("127.0.0.1", 12345)

                # Valid video_frame with data URL
                valid_message = json.dumps(
                    {
                        "type": "video_frame",
                        "frame": "data:image/jpeg;base64,validbase64data",
                        "prompt": "Analyze video",
                        "source": "video",
                    }
                )

                # Mock the async iteration properly
                websocket.__aiter__ = lambda self: async_messages()

                # Create async iterator
                async def async_messages():
                    yield valid_message

                # Add timeout to prevent hanging
                websocket.close = AsyncMock()
                try:
                    await asyncio.wait_for(server.handle_client(websocket, "/"), timeout=2.0)
                except asyncio.TimeoutError:
                    pass
                except Exception:
                    pass  # Ignore other exceptions

                # Should have processed the frame
                assert len(captured_data) == 1
                assert captured_data[0]["type"] == "image"
                assert captured_data[0]["content"] == "data:image/jpeg;base64,validbase64data"
                assert captured_data[0]["prompt"] == "Analyze video"
                assert captured_data[0]["source"] == "video"


class TestDefaultsAndInitialization:
    """Test default values and initialization."""

    def test_default_configuration(self):
        """Test default configuration values."""
        with mock_mlx_models_context():
            server = create_server()

            # Check defaults
            assert server.config["candidateCount"] == 1
            assert server.config["maxTokens"] == 200
            assert server.config["temperature"] == 0.7
            assert server.config["topP"] == 1.0
            assert server.config["topK"] == 50
            assert server.config["repetitionPenalty"] == 1.0
            assert server.config["repetitionContextSize"] == 20
            assert server.config["responseModalities"] == ["TEXT"]

    def test_custom_initialization(self):
        """Test custom initialization parameters."""
        with mock_mlx_models_context():
            server = create_server(
                model_name="custom/model",
                port=9999,
                host="localhost",
                debug=True,
                max_concurrent_inference=4,
                enable_response_cache=False,
                cache_size=200,
            )

            assert server.model_name == "custom/model"
            assert server.port == 9999
            assert server.host == "localhost"
            assert server.debug is True
            assert server.max_concurrent_inference == 4
            assert server.enable_response_cache is False
            assert server.response_cache is None  # Cache disabled


class TestEdgeCaseValidation:
    """Test edge case validation scenarios."""

    @pytest.mark.asyncio
    async def test_empty_config_update(self):
        """Test handling of empty config update."""
        with mock_mlx_models_context():
            server = create_server()

            websocket = AsyncMock()

            # Empty config update
            config_update = {}
            await server._handle_config(config_update, websocket)

            # Should send response with no updates
            sent_msg = json.loads(websocket.send.call_args[0][0])
            assert sent_msg["type"] == "config_updated"
            assert sent_msg["updated_fields"] == {}
            assert "current_config" in sent_msg

    @pytest.mark.asyncio
    async def test_unknown_config_fields(self):
        """Test handling of unknown configuration fields."""
        with mock_mlx_models_context():
            server = create_server()

            websocket = AsyncMock()

            # Config with unknown fields
            config_update = {"temperature": 0.8, "unknownField": "value", "anotherUnknown": 123}

            await server._handle_config(config_update, websocket)

            # Should only update known fields
            sent_msg = json.loads(websocket.send.call_args[0][0])
            assert sent_msg["updated_fields"] == {"temperature": 0.8}
            assert "unknownField" not in sent_msg["updated_fields"]

    @pytest.mark.asyncio
    async def test_unicode_in_messages(self):
        """Test handling of unicode characters in messages."""
        # Set up queue to capture data
        captured_data = []

        # Create mock queue instance
        mock_queue_instance = MagicMock()
        mock_queue_instance.full.return_value = False
        mock_queue_instance.put.side_effect = lambda x: captured_data.append(x)
        mock_queue_instance.get.side_effect = queue.Empty
        mock_queue_instance.empty.return_value = True
        mock_queue_instance.qsize.return_value = 0

        # Remove mlx_websockets modules if they're already imported
        modules_to_remove = [k for k in sys.modules.keys() if k.startswith("mlx_websockets")]
        for module in modules_to_remove:
            del sys.modules[module]

        # Patch Queue before importing server
        with patch("queue.Queue", return_value=mock_queue_instance):
            with mock_mlx_models_context():
                # Import server after patching Queue
                from mlx_websockets.server import MLXStreamingServer, _import_dependencies

                _import_dependencies(debug=True)

                server = MLXStreamingServer(debug=True)

                websocket = AsyncMock()
                websocket.remote_address = ("127.0.0.1", 12345)

                # Message with unicode
                message = json.dumps({"type": "text_input", "content": "Hello 你好 مرحبا 🌍"})
                # Mock the async iteration properly
                websocket.__aiter__ = lambda self: async_messages()

                # Create async iterator
                async def async_messages():
                    yield message

                # Add timeout and close method
                websocket.close = AsyncMock()
                try:
                    await asyncio.wait_for(server.handle_client(websocket, "/"), timeout=2.0)
                except asyncio.TimeoutError:
                    pass
                except Exception:
                    pass  # Ignore other exceptions

                # Should handle unicode properly
                assert len(captured_data) == 1
                assert captured_data[0]["content"] == "Hello 你好 مرحبا 🌍"
                assert captured_data[0]["type"] == "text"
