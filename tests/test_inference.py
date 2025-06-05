"""
Comprehensive model inference tests for MLX WebSocket Streaming Server
"""

import asyncio
import base64
import io
import json
import threading
import time
from unittest.mock import AsyncMock

import pytest
from PIL import Image

from .test_helpers import ServerTestContext, mock_generate_streaming


class TestTextModelInference:
    """Test text model inference capabilities"""

    def test_simple_text_generation(self):
        """Test basic text generation"""
        with ServerTestContext() as ctx:
            # Mock token generation
            ctx.mocks["generate"].side_effect = lambda *args, **kwargs: iter(
                ["Hello", " ", "world", "!"]
            )

            # Create websocket and collect messages
            messages = []
            websocket = ctx.create_websocket(messages)

            # Process text
            ctx.process_text(websocket, "Tell me a joke")

            # Wait for async operations
            time.sleep(0.5)

            # Parse messages
            parsed_messages = [json.loads(msg) for msg in messages]

            # Verify response structure
            assert len(parsed_messages) >= 6  # start + 4 tokens + complete
            assert parsed_messages[0]["type"] == "response_start"
            assert parsed_messages[0]["input_type"] == "text"

            # Check tokens
            tokens = [msg["content"] for msg in parsed_messages if msg["type"] == "token"]
            assert tokens == ["Hello", " ", "world", "!"]

            # Check completion
            assert parsed_messages[-1]["type"] == "response_complete"
            assert parsed_messages[-1]["full_text"] == "Hello world!"
            assert parsed_messages[-1]["input_type"] == "text"
            assert "inference_time" in parsed_messages[-1]

    def test_text_generation_with_context(self):
        """Test text generation with context"""
        with ServerTestContext() as ctx:
            captured_prompt = None

            def capture_prompt(*args, **kwargs):
                nonlocal captured_prompt
                captured_prompt = kwargs.get("prompt", "")
                return iter(["Context", " ", "aware", " ", "response"])

            ctx.mocks["generate"].side_effect = capture_prompt

            # Create websocket
            messages = []
            websocket = ctx.create_websocket(messages)

            # Process text with context
            ctx.process_text(
                websocket, "What's the weather like?", context="User is in San Francisco"
            )

            # Wait for completion
            time.sleep(0.5)

            # Verify context was included in prompt
            assert captured_prompt is not None
            assert "Context: User is in San Francisco" in captured_prompt
            assert "What's the weather like?" in captured_prompt

    def test_text_generation_parameters(self):
        """Test text generation with custom parameters"""
        with ServerTestContext() as ctx:
            captured_params = {}

            def capture_params(*args, **kwargs):
                nonlocal captured_params
                captured_params = kwargs
                return iter(["Test"])

            ctx.mocks["generate"].side_effect = capture_params

            # Update config
            with ctx.server.config_lock:
                ctx.server.config["temperature"] = 0.9
                ctx.server.config["maxOutputTokens"] = 500
                ctx.server.config["topP"] = 0.95
                ctx.server.config["topK"] = 40
                ctx.server.config["presencePenalty"] = 0.5
                ctx.server.config["frequencyPenalty"] = 0.3

            # Process text
            messages = []
            websocket = ctx.create_websocket(messages)
            ctx.process_text(websocket, "Test prompt")

            # Wait for completion
            time.sleep(0.5)

            # Verify parameters were passed correctly
            assert captured_params["temperature"] == 0.9
            assert captured_params["max_tokens"] == 500
            assert captured_params["top_p"] == 0.95
            assert captured_params["top_k"] == 40
            # Repetition penalty should be calculated from presence/frequency penalties
            assert captured_params["repetition_penalty"] > 1.0
            assert captured_params["stream"] is True

    def test_long_text_generation(self):
        """Test generation of long text responses"""
        with ServerTestContext() as ctx:
            # Generate a long response
            long_response = ["This ", "is ", "a ", "very ", "long ", "response "] * 20
            ctx.mocks["generate"].return_value = iter(long_response)

            # Process text
            messages = []
            websocket = ctx.create_websocket(messages)
            ctx.process_text(websocket, "Generate a long story")

            # Wait for completion
            time.sleep(1.0)

            # Parse messages
            parsed_messages = [json.loads(msg) for msg in messages]

            # Verify all tokens were sent
            tokens = [msg["content"] for msg in parsed_messages if msg["type"] == "token"]
            assert len(tokens) == len(long_response)

            # Verify final response
            complete_msg = next(
                msg for msg in parsed_messages if msg["type"] == "response_complete"
            )
            assert complete_msg["full_text"] == "".join(long_response)


class TestVisionModelInference:
    """Test vision model inference capabilities"""

    def create_test_image(self, size=(100, 100), color="red"):
        """Helper to create test images"""
        img = Image.new("RGB", size, color=color)
        img_buffer = io.BytesIO()
        img.save(img_buffer, format="PNG")
        img_buffer.seek(0)
        return img, base64.b64encode(img_buffer.getvalue()).decode()

    def test_image_inference_basic(self):
        """Test basic image inference"""
        with ServerTestContext() as ctx:
            ctx.mocks["generate"].return_value = iter(["A", " ", "red", " ", "square"])

            # Create test image
            _, img_base64 = self.create_test_image()

            # Create websocket
            messages = []
            websocket = ctx.create_websocket(messages)

            # Process image
            ctx.server._process_image(
                {
                    "type": "image",
                    "timestamp": time.time(),
                    "content": f"data:image/png;base64,{img_base64}",
                    "prompt": "What do you see?",
                    "source": "test",
                },
                websocket,
                ctx.loop,
                ("127.0.0.1", 12345),
                threading.Event(),
            )

            # Wait for completion
            time.sleep(0.5)

            # Parse messages
            parsed_messages = [json.loads(msg) for msg in messages]

            # Verify response
            assert parsed_messages[0]["type"] == "response_start"
            assert parsed_messages[0]["input_type"] == "image"

            tokens = [msg["content"] for msg in parsed_messages if msg["type"] == "token"]
            assert tokens == ["A", " ", "red", " ", "square"]

            assert parsed_messages[-1]["type"] == "response_complete"
            assert parsed_messages[-1]["full_text"] == "A red square"

    def test_image_resizing(self):
        """Test that large images are resized"""
        with ServerTestContext() as ctx:
            captured_image = None

            def capture_image(*args, **kwargs):
                nonlocal captured_image
                captured_image = kwargs.get("image")
                return iter(["Resized"])

            ctx.mocks["generate"].side_effect = capture_image

            # Create large test image
            _, img_base64 = self.create_test_image(size=(2000, 2000))

            # Process image
            messages = []
            websocket = ctx.create_websocket(messages)

            import threading

            ctx.server._process_image(
                {
                    "type": "image",
                    "timestamp": time.time(),
                    "content": f"data:image/png;base64,{img_base64}",
                    "prompt": "Describe",
                    "source": "test",
                },
                websocket,
                ctx.loop,
                ("127.0.0.1", 12345),
                threading.Event(),
            )

            # Wait for completion
            time.sleep(0.5)

            # Verify image was resized
            assert captured_image is not None
            assert max(captured_image.size) <= 768

    def test_different_image_formats(self):
        """Test different image formats (PNG, JPEG, etc.)"""
        with ServerTestContext() as ctx:
            ctx.mocks["generate"].return_value = iter(["Image", " ", "processed"])

            # Test different formats
            formats = ["PNG", "JPEG", "BMP"]

            for fmt in formats:
                img = Image.new("RGB", (100, 100), color="blue")
                img_buffer = io.BytesIO()

                # BMP doesn't support all save options
                if fmt == "BMP":
                    img.save(img_buffer, format=fmt)
                else:
                    img.save(img_buffer, format=fmt, quality=95)

                img_base64 = base64.b64encode(img_buffer.getvalue()).decode()

                # Process image
                messages = []
                websocket = ctx.create_websocket(messages)

                import threading

                ctx.server._process_image(
                    {
                        "type": "image",
                        "timestamp": time.time(),
                        "content": f"data:image/{fmt.lower()};base64,{img_base64}",
                        "prompt": f"Describe this {fmt} image",
                        "source": "test",
                    },
                    websocket,
                    ctx.loop,
                    ("127.0.0.1", 12345),
                    threading.Event(),
                )

                # Wait for completion
                time.sleep(0.5)

                # Verify each format was processed
                parsed_messages = [json.loads(msg) for msg in messages]
                assert any(msg["type"] == "response_complete" for msg in parsed_messages)

    def test_image_with_custom_max_tokens(self):
        """Test image inference with custom max tokens"""
        with ServerTestContext() as ctx:
            captured_max_tokens = None

            def capture_max_tokens(*args, **kwargs):
                nonlocal captured_max_tokens
                captured_max_tokens = kwargs.get("max_tokens")
                return iter(["Short"])

            ctx.mocks["generate"].side_effect = capture_max_tokens

            # Set custom max tokens for images
            with ctx.server.config_lock:
                ctx.server.max_tokens_image = 50

            # Create test image
            _, img_base64 = self.create_test_image()

            # Process image
            messages = []
            websocket = ctx.create_websocket(messages)

            import threading

            ctx.server._process_image(
                {
                    "type": "image",
                    "timestamp": time.time(),
                    "content": f"data:image/png;base64,{img_base64}",
                    "prompt": "Brief description",
                    "source": "test",
                },
                websocket,
                ctx.loop,
                ("127.0.0.1", 12345),
                threading.Event(),
            )

            # Wait for completion
            time.sleep(0.5)

            # Verify custom max tokens was used
            assert captured_max_tokens == 50


class TestStreamingBehavior:
    """Test streaming-specific behaviors"""

    def test_concurrent_streams(self):
        """Test handling multiple concurrent streams"""
        with ServerTestContext() as ctx:
            # Create different responses for different clients
            def create_generator(prompt):
                if "client1" in prompt:
                    return iter(["Client1", " ", "response"])
                elif "client2" in prompt:
                    return iter(["Client2", " ", "response"])
                else:
                    return iter(["Default", " ", "response"])

            ctx.mocks["generate"].side_effect = lambda *args, **kwargs: create_generator(
                kwargs.get("prompt", "")
            )

            # Create websockets for two clients
            messages1 = []
            messages2 = []
            ws1 = ctx.create_websocket(messages1)
            ws2 = ctx.create_websocket(messages2)

            # Process both clients concurrently
            import threading

            def process_client1():
                ctx.process_text(ws1, "Request from client1", ("127.0.0.1", 12345))

            def process_client2():
                ctx.process_text(ws2, "Request from client2", ("127.0.0.1", 12346))

            thread1 = threading.Thread(target=process_client1)
            thread2 = threading.Thread(target=process_client2)

            thread1.start()
            thread2.start()

            thread1.join(timeout=5.0)
            thread2.join(timeout=5.0)

            # Wait for async operations
            time.sleep(0.5)

            # Verify each client got their unique response
            parsed1 = [json.loads(msg) for msg in messages1]
            parsed2 = [json.loads(msg) for msg in messages2]

            tokens1 = [msg["content"] for msg in parsed1 if msg["type"] == "token"]
            tokens2 = [msg["content"] for msg in parsed2 if msg["type"] == "token"]

            assert "Client1" in "".join(tokens1)
            assert "Client2" in "".join(tokens2)

    def test_stream_interruption(self):
        """Test handling stream interruption"""
        with ServerTestContext() as ctx:
            # Create a long generator that will be interrupted
            def long_generator():
                for i in range(100):
                    yield f"Token{i} "

            ctx.mocks["generate"].return_value = long_generator()

            # Create websocket that fails after a few messages
            messages = []

            class InterruptingWebSocket:
                def __init__(self):
                    self.remote_address = ("127.0.0.1", 12345)
                    self.closed = False
                    self.send_count = 0

                async def send(self, message):
                    self.send_count += 1
                    if self.send_count > 5:
                        import websockets

                        raise websockets.exceptions.ConnectionClosed(None, None)
                    messages.append(message)

                async def close(self):
                    self.closed = True

            websocket = InterruptingWebSocket()

            # Process text
            ctx.server._process_text(
                {
                    "type": "text",
                    "timestamp": time.time(),
                    "content": "Generate long text",
                    "context": "",
                    "prompt": "Generate long text",
                },
                websocket,
                ctx.loop,
                ("127.0.0.1", 12345),
                threading.Event(),
            )

            # Wait for completion
            time.sleep(0.5)

            # Verify stream was interrupted gracefully
            assert len(messages) <= 10  # Should stop early
            parsed = [json.loads(msg) for msg in messages]
            assert not any(msg["type"] == "error" for msg in parsed)


class TestErrorHandlingAndEdgeCases:
    """Test error handling and edge cases"""

    def test_invalid_image_data(self):
        """Test handling of invalid image data"""
        with ServerTestContext() as ctx:
            # Process invalid image
            messages = []
            websocket = ctx.create_websocket(messages)

            import threading

            ctx.server._process_image(
                {
                    "type": "image",
                    "timestamp": time.time(),
                    "content": "data:image/png;base64,INVALID_BASE64_DATA",
                    "prompt": "Describe",
                    "source": "test",
                },
                websocket,
                ctx.loop,
                ("127.0.0.1", 12345),
                threading.Event(),
            )

            # Wait for completion
            time.sleep(0.5)

            # Verify error was sent
            parsed = [json.loads(msg) for msg in messages]
            error_msgs = [msg for msg in parsed if msg["type"] == "error"]
            assert len(error_msgs) > 0

    def test_model_error_handling(self):
        """Test handling of model errors"""
        with ServerTestContext() as ctx:
            # Make generate raise an error
            ctx.mocks["generate"].side_effect = Exception("Model error")

            # Process text
            messages = []
            websocket = ctx.create_websocket(messages)
            ctx.process_text(websocket, "Test prompt")

            # Wait for completion
            time.sleep(0.5)

            # Verify error was sent
            parsed = [json.loads(msg) for msg in messages]
            error_msgs = [msg for msg in parsed if msg["type"] == "error"]
            assert len(error_msgs) > 0
            # The server tries vision API first, then falls back to text API
            # Since we don't mock text_generate, we get this error
            assert (
                "Text-only model API not available and vision API failed" in error_msgs[0]["error"]
            )

    def test_empty_input_handling(self):
        """Test handling of empty inputs"""
        with ServerTestContext() as ctx:
            # Test empty text input
            messages = []
            websocket = ctx.create_websocket(messages)

            # Send empty text - should process normally with empty prompt
            ctx.process_text(websocket, "")

            # Wait for processing
            time.sleep(0.5)

            # Verify response was sent (empty prompt is valid)
            parsed = [json.loads(msg) for msg in messages]
            assert any(msg["type"] == "response_start" for msg in parsed)
            assert any(msg["type"] == "response_complete" for msg in parsed)


class TestPerformanceAndLoad:
    """Test performance and load handling"""

    def test_rapid_message_handling(self):
        """Test handling rapid message bursts"""
        with ServerTestContext() as ctx:
            ctx.mocks["generate"].return_value = iter(["Quick", " ", "response"])

            # Send many messages rapidly
            messages = []
            websocket = ctx.create_websocket(messages)

            for i in range(20):
                ctx.process_text(websocket, f"Message {i}")
                time.sleep(0.01)  # Small delay between messages

            # Wait for processing
            time.sleep(2.0)

            # Verify messages were processed
            parsed = [json.loads(msg) for msg in messages]
            completions = [msg for msg in parsed if msg["type"] == "response_complete"]

            assert len(completions) > 0  # Some messages processed

    def test_memory_efficient_streaming(self):
        """Test memory efficiency during streaming"""
        with ServerTestContext() as ctx:
            # Generate a very long response
            token_count = [0]

            def counting_generator():
                while token_count[0] < 1000:
                    token_count[0] += 1
                    yield f"Token{token_count[0]} "

            ctx.mocks["generate"].return_value = counting_generator()

            # Create websocket that stops after many tokens
            messages = []

            class LimitingWebSocket:
                def __init__(self):
                    self.remote_address = ("127.0.0.1", 12345)
                    self.closed = False
                    self.token_count = 0

                async def send(self, message):
                    data = json.loads(message)
                    if data["type"] == "token":
                        self.token_count += 1
                        if self.token_count >= 1000:
                            import websockets

                            raise websockets.exceptions.ConnectionClosed(None, None)
                    messages.append(message)

                async def close(self):
                    self.closed = True

            websocket = LimitingWebSocket()

            # Process text
            import threading

            ctx.server._process_text(
                {
                    "type": "text",
                    "timestamp": time.time(),
                    "content": "Generate infinite text",
                    "context": "",
                    "prompt": "Generate infinite text",
                },
                websocket,
                ctx.loop,
                ("127.0.0.1", 12345),
                threading.Event(),
            )

            # Wait for completion
            time.sleep(2.0)

            # Verify streaming stopped appropriately
            assert websocket.token_count >= 1000

            # Verify generator was cleaned up
            with ctx.server.clients_lock:
                assert len(ctx.server.client_generators.get(("127.0.0.1", 12345), [])) == 0

    def test_concurrent_client_limits(self):
        """Test handling many concurrent clients"""
        with ServerTestContext() as ctx:
            # Create many clients
            num_clients = 50
            clients = []

            for i in range(num_clients):
                client_id = ("127.0.0.1", 12345 + i)

                with ctx.server.clients_lock:
                    import queue
                    import threading

                    ctx.server.client_queues[client_id] = queue.Queue()
                    ctx.server.client_stop_events[client_id] = threading.Event()
                    ctx.server.client_frame_counts[client_id] = 0
                    ctx.server.client_generators[client_id] = []

                clients.append(client_id)

            # Verify all clients were added
            with ctx.server.clients_lock:
                assert len(ctx.server.client_queues) == num_clients

            # Clean up half the clients
            for i in range(num_clients // 2):
                client_id = clients[i]
                with ctx.server.clients_lock:
                    ctx.server.client_queues.pop(client_id, None)
                    ctx.server.client_stop_events.pop(client_id, None)
                    ctx.server.client_frame_counts.pop(client_id, None)
                    ctx.server.client_generators.pop(client_id, None)

            # Verify cleanup
            with ctx.server.clients_lock:
                assert len(ctx.server.client_queues) == num_clients // 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
