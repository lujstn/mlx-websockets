"""Error handling tests for MLX WebSocket server."""

import asyncio
import base64
import io
import json
import time
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from mlx_websockets.exceptions import (
    ConfigurationError,
    ImageProcessingError,
    MessageProcessingError,
    ModelLoadError,
    NetworkError,
    ResourceError,
    TextGenerationError,
)

from .test_helpers import mock_mlx_models, mock_mlx_models_context


def create_server(**kwargs):
    """Helper to create server with proper mocking."""
    from mlx_websockets.server import MLXStreamingServer

    return MLXStreamingServer(**kwargs)


class TestModelLoadingErrors:
    """Test error handling during model loading."""

    def test_model_not_found_error(self):
        """Test handling of model not found error."""
        with mock_mlx_models_context():
            from mlx_websockets import server
            from mlx_websockets.server import MLXStreamingServer

            with patch.object(server, "load", side_effect=FileNotFoundError("Model not found")):
                with pytest.raises(ModelLoadError) as exc_info:
                    MLXStreamingServer(model_name="nonexistent/model")

                assert "not found" in str(exc_info.value)
                assert "Please check the model name" in str(exc_info.value)

    def test_import_error_during_load(self):
        """Test handling of import errors during model loading."""
        with mock_mlx_models_context():
            from mlx_websockets import server
            from mlx_websockets.server import MLXStreamingServer

            with patch.object(server, "load", side_effect=ImportError("Missing dependency")):
                with pytest.raises(ModelLoadError) as exc_info:
                    MLXStreamingServer()

                assert "Missing required dependencies" in str(exc_info.value)
                assert "mlx-vlm" in str(exc_info.value)

    def test_memory_error_during_load(self):
        """Test handling of memory errors during model loading."""
        with mock_mlx_models_context():
            from mlx_websockets import server
            from mlx_websockets.server import MLXStreamingServer

            with patch.object(server, "load", side_effect=MemoryError("Out of memory")):
                with pytest.raises(ModelLoadError) as exc_info:
                    MLXStreamingServer()

                assert "Insufficient memory" in str(exc_info.value)
                assert "smaller model" in str(exc_info.value)

    def test_generic_error_during_load(self):
        """Test handling of generic errors during model loading."""
        with mock_mlx_models_context():
            from mlx_websockets import server
            from mlx_websockets.server import MLXStreamingServer

            with patch.object(server, "load", side_effect=RuntimeError("Unexpected error")):
                with pytest.raises(ModelLoadError) as exc_info:
                    MLXStreamingServer()

                assert "Failed to load model" in str(exc_info.value)
                assert "Unexpected error" in str(exc_info.value)


class TestWebSocketMessageErrors:
    """Test error handling for WebSocket message processing."""

    @pytest.mark.asyncio
    async def test_invalid_json_message(self):
        """Test handling of invalid JSON in WebSocket messages."""
        with mock_mlx_models_context():
            server = create_server()

            websocket = AsyncMock()
            websocket.remote_address = ("127.0.0.1", 12345)
            websocket.__aiter__.return_value = ["invalid json{"]

            await server.handle_client(websocket, "/")

            # Should send error response
            sent_messages = [call[0][0] for call in websocket.send.call_args_list]
            assert any("error" in msg and "Invalid message format" in msg for msg in sent_messages)

    @pytest.mark.asyncio
    async def test_missing_image_content(self):
        """Test handling of missing image content in messages."""
        with mock_mlx_models_context():
            server = create_server()

            websocket = AsyncMock()
            websocket.remote_address = ("127.0.0.1", 12345)

            # Message with missing image content
            message = json.dumps(
                {
                    "type": "image_input",
                    "prompt": "Describe this",
                    # Missing "image" field
                }
            )
            websocket.__aiter__.return_value = [message]

            await server.handle_client(websocket, "/")

            # Should send error response
            sent_messages = [call[0][0] for call in websocket.send.call_args_list]
            assert any(
                "error" in msg and "Missing image/frame content" in msg for msg in sent_messages
            )

    @pytest.mark.asyncio
    async def test_oversized_image_content(self):
        """Test handling of oversized image content."""
        with mock_mlx_models_context():
            server = create_server()

            websocket = AsyncMock()
            websocket.remote_address = ("127.0.0.1", 12345)

            # Create oversized content (>10MB)
            oversized_content = "x" * (11 * 1024 * 1024)
            message = json.dumps(
                {"type": "image_input", "image": oversized_content, "prompt": "Describe this"}
            )
            websocket.__aiter__.return_value = [message]

            await server.handle_client(websocket, "/")

            # Should send error response
            sent_messages = [call[0][0] for call in websocket.send.call_args_list]
            assert any(
                "error" in msg and "too large" in msg and "10MB" in msg for msg in sent_messages
            )

    @pytest.mark.asyncio
    async def test_missing_text_content(self):
        """Test handling of missing text content in messages."""
        with mock_mlx_models_context():
            server = create_server()

            websocket = AsyncMock()
            websocket.remote_address = ("127.0.0.1", 12345)

            # Message with missing text content
            message = json.dumps(
                {
                    "type": "text_input"
                    # Missing "content" field
                }
            )
            websocket.__aiter__.return_value = [message]

            await server.handle_client(websocket, "/")

            # Should send error response
            sent_messages = [call[0][0] for call in websocket.send.call_args_list]
            assert any("error" in msg and "Missing text content" in msg for msg in sent_messages)

    @pytest.mark.asyncio
    async def test_oversized_text_content(self):
        """Test handling of oversized text content."""
        with mock_mlx_models_context():
            server = create_server()

            websocket = AsyncMock()
            websocket.remote_address = ("127.0.0.1", 12345)

            # Create oversized text content (>1MB)
            oversized_content = "x" * (2 * 1024 * 1024)
            message = json.dumps({"type": "text_input", "content": oversized_content})
            websocket.__aiter__.return_value = [message]

            await server.handle_client(websocket, "/")

            # Should send error response
            sent_messages = [call[0][0] for call in websocket.send.call_args_list]
            assert any(
                "error" in msg and "too large" in msg and "1MB" in msg for msg in sent_messages
            )


class TestImageProcessingErrors:
    """Test error handling during image processing."""

    def test_invalid_base64_image(self):
        """Test handling of invalid base64 image data."""
        with mock_mlx_models_context():
            server = create_server()

            websocket = Mock()
            loop = asyncio.new_event_loop()
            client_id = ("127.0.0.1", 12345)
            stop_event = Mock(is_set=Mock(return_value=False))

            # Invalid base64 data
            data = {
                "type": "image",
                "content": "invalid-base64-data!!!",
                "prompt": "Describe this",
                "timestamp": time.time(),
            }

            with patch.object(server, "_safe_send") as mock_send:
                server._process_image(data, websocket, loop, client_id, stop_event)

                # Should send error message
                error_calls = [
                    call
                    for call in mock_send.call_args_list
                    if "error" in call[0][1] and "Invalid image data" in call[0][1]
                ]
                assert len(error_calls) > 0

    def test_corrupted_image_data(self):
        """Test handling of corrupted image data."""
        with mock_mlx_models_context():
            server = create_server()

            websocket = Mock()
            loop = asyncio.new_event_loop()
            client_id = ("127.0.0.1", 12345)
            stop_event = Mock(is_set=Mock(return_value=False))

            # Valid base64 but not a valid image
            corrupted_data = base64.b64encode(b"not an image").decode()
            data = {
                "type": "image",
                "content": corrupted_data,
                "prompt": "Describe this",
                "timestamp": time.time(),
            }

            # Import the server module to patch its Image global
            from mlx_websockets import server as server_module

            with patch.object(server_module.Image, "open", side_effect=Exception("Invalid image")):
                with patch.object(server, "_safe_send") as mock_send:
                    server._process_image(data, websocket, loop, client_id, stop_event)

                    # Should send error message
                    error_calls = [
                        call
                        for call in mock_send.call_args_list
                        if "error" in call[0][1] and "Failed to process image" in call[0][1]
                    ]
                    assert len(error_calls) > 0

    def test_memory_error_during_image_processing(self):
        """Test handling of memory errors during image processing."""
        with mock_mlx_models_context():
            server = create_server()

            websocket = Mock()
            loop = asyncio.new_event_loop()
            client_id = ("127.0.0.1", 12345)
            stop_event = Mock(is_set=Mock(return_value=False))

            # Create a simple valid image
            img = MagicMock()
            img_bytes = io.BytesIO()
            img.save(img_bytes, format="PNG")
            valid_image_data = base64.b64encode(img_bytes.getvalue()).decode()

            data = {
                "type": "image",
                "content": f"data:image/png;base64,{valid_image_data}",
                "prompt": "Describe this",
                "timestamp": time.time(),
            }

            # Import the server module to patch its Image global
            from mlx_websockets import server as server_module

            with patch.object(
                server_module.Image, "open", side_effect=MemoryError("Out of memory")
            ):
                with patch.object(server, "_safe_send") as mock_send:
                    server._process_image(data, websocket, loop, client_id, stop_event)

                    # Should send memory error message
                    error_calls = [
                        call
                        for call in mock_send.call_args_list
                        if "error" in call[0][1] and "Out of memory" in call[0][1]
                    ]
                    assert len(error_calls) > 0


class TestTextProcessingErrors:
    """Test error handling during text processing."""

    def test_text_generation_not_available(self):
        """Test handling when text generation is not available."""
        with mock_mlx_models_context():
            server = create_server()

            websocket = Mock()
            loop = asyncio.new_event_loop()
            client_id = ("127.0.0.1", 12345)
            stop_event = Mock(is_set=Mock(return_value=False))

            data = {"type": "text", "content": "Hello", "prompt": "Hello", "timestamp": time.time()}

            # Make generate fail with AttributeError
            from mlx_websockets import server as server_module

            with patch.object(
                server_module, "generate", side_effect=AttributeError("No text support")
            ):
                with patch.object(server_module, "text_generate", None):
                    with patch.object(server, "_safe_send") as mock_send:
                        server._process_text(data, websocket, loop, client_id, stop_event)

                        # Should send error about text generation or failed to process
                        error_calls = [
                            call
                            for call in mock_send.call_args_list
                            if "error" in call[0][1]
                            and (
                                "Text generation" in call[0][1] or "Failed to process" in call[0][1]
                            )
                        ]
                        assert len(error_calls) > 0

    def test_text_validation_error(self):
        """Test handling of text validation errors."""
        with mock_mlx_models_context():
            server = create_server()

            websocket = Mock()
            loop = asyncio.new_event_loop()
            client_id = ("127.0.0.1", 12345)
            stop_event = Mock(is_set=Mock(return_value=False))

            data = {"type": "text", "content": "Test", "prompt": "Test", "timestamp": time.time()}

            # Mock _safe_send to return True
            mock_send_calls = []

            def mock_safe_send(ws, msg, loop):
                mock_send_calls.append(msg)
                return True

            server._safe_send = mock_safe_send

            # Import the server module to patch generate
            from mlx_websockets import server as server_module

            with patch.object(server_module, "generate", side_effect=ValueError("Invalid input")):
                server._process_text(data, websocket, loop, client_id, stop_event)

                # Should send validation error
                error_messages = []
                for msg in mock_send_calls:
                    try:
                        parsed = json.loads(msg)
                        if parsed.get("type") == "error" and "Invalid text input" in parsed.get(
                            "error", ""
                        ):
                            error_messages.append(parsed)
                    except (json.JSONDecodeError, KeyError, TypeError):
                        pass

                assert (
                    len(error_messages) > 0
                ), f"Expected error message with 'Invalid text input', got: {mock_send_calls}"


class TestStreamingErrors:
    """Test error handling during response streaming."""

    def test_streaming_timeout(self):
        """Test handling of streaming timeout."""
        with mock_mlx_models_context():
            server = create_server()

            websocket = Mock()
            loop = asyncio.new_event_loop()
            client_id = ("127.0.0.1", 12345)
            stop_event = Mock(is_set=Mock(return_value=False))

            data = {"timestamp": time.time()}

            # Create a generator that yields tokens normally
            def normal_generator():
                yield "token1"
                yield "token2"

            # Mock time.time() to simulate timeout condition
            # Need to ensure the timeout check (time.time() - inference_start > 60) triggers
            # inference_start is set at the beginning of _stream_response
            # The timeout check happens on each token iteration
            with patch("time.time") as mock_time:
                # Set up time progression: start at 0, then return 61+ when timeout check happens
                mock_time.side_effect = [0, 61.5]  # Start time, then timeout check time
                with patch.object(server, "_safe_send") as mock_send:
                    mock_send.return_value = True  # Simulate successful sends

                    with pytest.raises(TimeoutError) as exc_info:
                        server._stream_response(
                            normal_generator(), websocket, loop, client_id, stop_event, data, "text"
                        )

                    # Verify the timeout error message
                    assert "Inference exceeded 60s timeout" in str(exc_info.value)

    def test_streaming_with_connection_closed(self):
        """Test handling when connection closes during streaming."""
        with mock_mlx_models_context():
            server = create_server()

            websocket = Mock()
            loop = asyncio.new_event_loop()
            client_id = ("127.0.0.1", 12345)
            stop_event = Mock(is_set=Mock(return_value=False))

            data = {"timestamp": time.time()}

            # Create a generator
            def token_generator():
                yield "token1"
                yield "token2"
                yield "token3"

            # Simulate connection closing after first token
            send_results = [True, False, False]  # First succeeds, rest fail
            with patch.object(server, "_safe_send", side_effect=send_results):
                # Add client generator tracking
                with server.clients_lock:
                    server.client_generators[client_id] = []

                generator = token_generator()
                server._stream_response(
                    generator, websocket, loop, client_id, stop_event, data, "text"
                )

                # Should stop streaming when connection fails
                # Generator should be cleaned up
                with server.clients_lock:
                    assert generator not in server.client_generators.get(client_id, [])


class TestProcessingThreadErrors:
    """Test error handling in processing threads."""

    def test_processing_thread_os_error(self):
        """Test handling of OS errors in processing thread."""
        with mock_mlx_models_context():
            server = create_server()

            websocket = Mock()
            client_queue = Mock()
            # Create a proper stop event mock that returns False once then True
            call_count = 0

            def is_set_side_effect():
                nonlocal call_count
                call_count += 1
                return call_count > 2  # False for first 2 calls, then True

            stop_event = Mock(is_set=Mock(side_effect=is_set_side_effect))
            loop = asyncio.new_event_loop()
            client_id = ("127.0.0.1", 12345)

            # Simulate OS error when getting from queue (only once, then Empty)
            from queue import Empty

            client_queue.get.side_effect = [OSError("Network error"), Empty()]
            client_queue.get_nowait.side_effect = Empty()  # For cleanup
            client_queue.empty.return_value = True

            with patch.object(server, "_safe_send") as mock_send:
                server._process_frames(websocket, client_queue, stop_event, loop, client_id)

                # Should send network error
                error_calls = [
                    call
                    for call in mock_send.call_args_list
                    if "error" in call[0][1] and "Network error" in call[0][1]
                ]
                assert len(error_calls) > 0

    def test_processing_thread_memory_error(self):
        """Test handling of memory errors in processing thread."""
        with mock_mlx_models_context():
            server = create_server()

            websocket = Mock()
            client_queue = Mock()
            # Create a proper stop event mock that returns False once then True
            call_count = 0

            def is_set_side_effect():
                nonlocal call_count
                call_count += 1
                return call_count > 2  # False for first 2 calls, then True

            stop_event = Mock(is_set=Mock(side_effect=is_set_side_effect))
            loop = asyncio.new_event_loop()
            client_id = ("127.0.0.1", 12345)

            # Simulate memory error (only once, then Empty)
            from queue import Empty

            client_queue.get.side_effect = [MemoryError("Out of memory"), Empty()]
            client_queue.get_nowait.side_effect = Empty()  # For cleanup
            client_queue.empty.return_value = True

            with patch.object(server, "_safe_send") as mock_send:
                server._process_frames(websocket, client_queue, stop_event, loop, client_id)

                # Should send memory error
                error_calls = [
                    call
                    for call in mock_send.call_args_list
                    if "error" in call[0][1] and "Out of memory" in call[0][1]
                ]
                assert len(error_calls) > 0

    def test_processing_thread_generic_error(self):
        """Test handling of generic errors in processing thread."""
        with mock_mlx_models_context():
            server = create_server()

            websocket = Mock()
            client_queue = Mock()
            # Create a proper stop event mock that returns False once then True
            call_count = 0

            def is_set_side_effect():
                nonlocal call_count
                call_count += 1
                return call_count > 2  # False for first 2 calls, then True

            stop_event = Mock(is_set=Mock(side_effect=is_set_side_effect))
            loop = asyncio.new_event_loop()
            client_id = ("127.0.0.1", 12345)

            # Simulate generic error (only once, then Empty)
            from queue import Empty

            client_queue.get.side_effect = [RuntimeError("Unexpected error"), Empty()]
            client_queue.get_nowait.side_effect = Empty()  # For cleanup
            client_queue.empty.return_value = True

            with patch.object(server, "_safe_send") as mock_send:
                server._process_frames(websocket, client_queue, stop_event, loop, client_id)

                # Should send generic error
                error_calls = [
                    call
                    for call in mock_send.call_args_list
                    if "error" in call[0][1]
                    and "Processing thread encountered an error" in call[0][1]
                ]
                assert len(error_calls) > 0


class TestConnectionErrors:
    """Test connection-related error handling."""

    @pytest.mark.asyncio
    async def test_websocket_connection_closed(self):
        """Test handling of WebSocket connection closed."""
        with mock_mlx_models_context():
            import websockets.exceptions

            server = create_server()

            websocket = AsyncMock()
            websocket.remote_address = ("127.0.0.1", 12345)
            websocket.__aiter__.side_effect = websockets.exceptions.ConnectionClosed(None, None)

            # Should handle gracefully without crashing
            await server.handle_client(websocket, "/")

    @pytest.mark.asyncio
    async def test_asyncio_cancelled_error(self):
        """Test handling of cancelled async tasks."""
        with mock_mlx_models_context():
            server = create_server()

            websocket = AsyncMock()
            websocket.remote_address = ("127.0.0.1", 12345)
            websocket.__aiter__.side_effect = asyncio.CancelledError()

            # Should re-raise CancelledError
            with pytest.raises(asyncio.CancelledError):
                await server.handle_client(websocket, "/")

    @pytest.mark.asyncio
    async def test_generic_connection_error(self):
        """Test handling of generic connection errors."""
        with mock_mlx_models_context():
            server = create_server()

            websocket = AsyncMock()
            websocket.remote_address = ("127.0.0.1", 12345)
            websocket.__aiter__.side_effect = Exception("Connection failed")

            # Should handle gracefully
            await server.handle_client(websocket, "/")
