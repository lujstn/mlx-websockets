"""
Tests for MLX WebSocket Streaming Server
"""

import asyncio
import base64
import io
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from PIL import Image

from .test_helpers import (
    RealServerTestContext,
    mock_mlx_models,
    simulate_concurrent_clients,
    wait_for_thread_count,
)


class TestMLXStreamingServer:
    """Test the MLX Streaming Server core functionality"""

    @pytest.mark.asyncio
    async def test_server_initialization(self):
        """Test server initializes with correct defaults"""
        async with RealServerTestContext() as ctx:
            assert ctx.server.config["temperature"] == 0.7
            assert ctx.server.config["maxTokens"] == 200
            assert ctx.server.max_tokens_image == 100

    @pytest.mark.asyncio
    async def test_real_websocket_connection(self):
        """Test real WebSocket client-server connection"""
        async with RealServerTestContext() as ctx:
            client = ctx.create_client()
            await client.connect()

            # Send a simple message
            await client.send({"type": "text_input", "content": "Hello, server!", "context": ""})

            # Receive responses
            messages = await client.receive_all(timeout=5.0)

            # Verify we got proper responses
            assert len(messages) > 0
            assert any(msg["type"] == "response_start" for msg in messages)
            assert any(msg["type"] == "token" for msg in messages)
            assert any(msg["type"] == "response_complete" for msg in messages)

            await client.close()

    @pytest.mark.asyncio
    async def test_config_update_live(self):
        """Test configuration updates through WebSocket"""
        async with RealServerTestContext() as ctx:
            client = ctx.create_client()
            await client.connect()

            # Update config (using OpenAI-compatible parameter)
            await client.send({"type": "config", "temperature": 0.9, "maxOutputTokens": 500})

            # Give server time to process
            await asyncio.sleep(0.1)

            # Verify config was updated (converted to MLX-native parameter)
            assert ctx.server.config["temperature"] == 0.9
            assert ctx.server.config["maxTokens"] == 500

            await client.close()

    @pytest.mark.asyncio
    async def test_client_lifecycle(self):
        """Test full client connection lifecycle"""
        async with RealServerTestContext() as ctx:
            # Track initial state
            initial_clients = len(ctx.server.client_queues)

            # Connect client
            client = ctx.create_client()
            await client.connect()

            # Verify client is tracked
            await asyncio.sleep(0.1)
            assert len(ctx.server.client_queues) == initial_clients + 1

            # Send and receive
            await client.send({"type": "text_input", "content": "Test message"})

            messages = await client.receive_all()
            assert len(messages) > 0

            # Disconnect
            await client.close()

            # Verify cleanup
            await asyncio.sleep(0.5)
            assert len(ctx.server.client_queues) == initial_clients


class TestMessageHandling:
    """Test WebSocket message handling with real async operations"""

    @pytest.mark.asyncio
    async def test_text_processing_flow(self):
        """Test complete text processing flow"""
        async with RealServerTestContext() as ctx:
            messages = await ctx.test_full_message_flow(
                prompt="What is the capital of France?", expected_in_response="Generated response"
            )

            # Verify message structure
            start_msgs = [m for m in messages if m["type"] == "response_start"]
            token_msgs = [m for m in messages if m["type"] == "token"]
            complete_msgs = [m for m in messages if m["type"] == "response_complete"]

            assert len(start_msgs) == 1
            assert len(token_msgs) > 0
            assert len(complete_msgs) == 1

            # Verify complete message has all fields
            complete = complete_msgs[0]
            assert "full_text" in complete
            assert "inference_time" in complete
            assert "timestamp" in complete
            assert "input_type" in complete

    @pytest.mark.asyncio
    async def test_image_processing_flow(self):
        """Test image processing with real async flow"""
        async with RealServerTestContext() as ctx:
            client = ctx.create_client()
            await client.connect()

            # Create test image
            img = Image.new("RGB", (100, 100), color="red")
            img_buffer = io.BytesIO()
            img.save(img_buffer, format="PNG")
            img_base64 = base64.b64encode(img_buffer.getvalue()).decode()

            # Send image
            await client.send(
                {
                    "type": "image_input",
                    "image": f"data:image/png;base64,{img_base64}",
                    "prompt": "What's in this image?",
                }
            )

            # Receive responses
            messages = await client.receive_all()

            # Verify processing
            assert len(messages) > 0
            assert any(msg["type"] == "response_complete" for msg in messages)

            await client.close()

    @pytest.mark.asyncio
    async def test_streaming_tokens(self):
        """Test that tokens are streamed in real-time"""
        async with RealServerTestContext() as ctx:
            client = ctx.create_client()
            await client.connect()

            # Send message
            await client.send({"type": "text_input", "content": "Count to five"})

            # Collect tokens with timestamps
            token_times = []
            start_time = time.time()

            while True:
                msg = await client.receive(timeout=1.0)
                if not msg:
                    break

                if msg["type"] == "token":
                    token_times.append((msg["content"], time.time() - start_time))
                elif msg["type"] == "response_complete":
                    break

            # Verify tokens arrived over time (not all at once)
            assert len(token_times) > 2
            time_diffs = [
                token_times[i][1] - token_times[i - 1][1] for i in range(1, len(token_times))
            ]
            assert any(diff > 0.001 for diff in time_diffs)  # Some delay between tokens

            await client.close()


class TestConcurrency:
    """Test concurrent client handling"""

    @pytest.mark.asyncio
    async def test_multiple_concurrent_clients(self):
        """Test server handles multiple clients concurrently"""
        async with RealServerTestContext() as ctx:
            # Create multiple clients
            num_clients = 5
            clients = []

            for _i in range(num_clients):
                client = ctx.create_client()
                await client.connect()
                clients.append(client)

            # Send messages concurrently
            async def send_and_receive(client, client_id):
                await client.send(
                    {"type": "text_input", "content": f"Message from client {client_id}"}
                )
                messages = await client.receive_all()
                return len(messages) > 0

            # Run all clients concurrently
            results = await asyncio.gather(
                *[send_and_receive(client, i) for i, client in enumerate(clients)]
            )

            # All should succeed
            assert all(results)

            # Clean up
            for client in clients:
                await client.close()

    @pytest.mark.asyncio
    async def test_concurrent_message_ordering(self):
        """Test that each client's messages are processed in order"""
        async with RealServerTestContext() as ctx:
            client = ctx.create_client()
            await client.connect()

            # Send multiple messages quickly
            messages_sent = []
            for i in range(5):
                msg = f"Message {i}"
                messages_sent.append(msg)
                await client.send({"type": "text_input", "content": msg})

            # Collect all responses
            all_responses = []
            timeout = time.time() + 10

            while time.time() < timeout:
                msg = await client.receive(timeout=0.5)
                if not msg:
                    break
                all_responses.append(msg)

                # Count complete messages
                complete_count = sum(1 for m in all_responses if m["type"] == "response_complete")
                if complete_count == len(messages_sent):
                    break

            # Verify we got responses for all messages
            complete_messages = [m for m in all_responses if m["type"] == "response_complete"]
            assert len(complete_messages) == len(messages_sent)

            await client.close()

    @pytest.mark.asyncio
    @pytest.mark.timeout(20)  # Specific timeout for this test
    async def test_thread_pool_stress(self):
        """Test thread pool under load - focuses on functionality not cleanup timing"""
        async with RealServerTestContext() as ctx:
            # Simulate concurrent operations with fewer clients to reduce complexity
            responses = await simulate_concurrent_clients(
                ctx.actual_port, num_clients=3, messages_per_client=1
            )

            # Verify all clients got responses - this is the main test goal
            assert len(responses) == 3
            assert all(len(client_responses) > 0 for client_responses in responses)

            # Verify responses contain expected message types
            for client_responses in responses:
                message_types = [msg.get("type") for msg in client_responses]
                assert "response_start" in message_types
                assert "response_complete" in message_types


class TestErrorHandling:
    """Test error handling with real operations"""

    @pytest.mark.asyncio
    async def test_invalid_message_handling(self):
        """Test server handles invalid messages gracefully"""
        async with RealServerTestContext() as ctx:
            client = ctx.create_client()
            await client.connect()

            # Send invalid message
            await client.send({"type": "invalid_type", "data": "test"})

            # Send valid message after
            await client.send({"type": "text_input", "content": "Valid message"})

            # Should still get response for valid message
            messages = await client.receive_all()
            assert any(msg["type"] == "response_complete" for msg in messages)

            await client.close()

    @pytest.mark.asyncio
    async def test_client_disconnect_cleanup(self):
        """Test cleanup when client disconnects abruptly"""
        async with RealServerTestContext() as ctx:
            client = ctx.create_client()
            await client.connect()

            # Start processing
            await client.send(
                {"type": "text_input", "content": "Start processing this long text..."}
            )

            # Disconnect abruptly (without proper close)
            if client.websocket:
                await client.websocket.close(code=1011, reason="Abrupt disconnect")

            # Wait for cleanup
            await asyncio.sleep(1.0)

            # Verify server is still functional
            new_client = ctx.create_client()
            await new_client.connect()
            await new_client.send({"type": "text_input", "content": "Still working?"})

            messages = await new_client.receive_all()
            assert len(messages) > 0

            await new_client.close()

    @pytest.mark.asyncio
    async def test_generation_interruption(self):
        """Test stopping generation mid-stream"""
        async with RealServerTestContext() as ctx:
            client = ctx.create_client()
            await client.connect()

            # Send message
            await client.send({"type": "text_input", "content": "Generate a very long response"})

            # Receive a few tokens
            token_count = 0
            for _ in range(5):
                msg = await client.receive(timeout=1.0)
                if msg and msg["type"] == "token":
                    token_count += 1

            # Close connection mid-generation
            await client.close()

            # Verify we got some but not all tokens
            assert token_count > 0
            assert token_count < 20  # Should have interrupted before completion


class TestThreadSafety:
    """Test thread safety with real concurrent operations"""

    @pytest.mark.asyncio
    async def test_concurrent_config_updates(self):
        """Test concurrent configuration updates"""
        async with RealServerTestContext() as ctx:
            clients = []

            # Create multiple clients
            for _ in range(5):
                client = ctx.create_client()
                await client.connect()
                clients.append(client)

            # Update config concurrently from different clients
            async def update_config(client, temp):
                await client.send({"type": "config", "temperature": temp})

            # Send concurrent updates
            tasks = [update_config(client, 0.5 + i * 0.1) for i, client in enumerate(clients)]
            await asyncio.gather(*tasks)

            # Wait for processing
            await asyncio.sleep(0.5)

            # Config should have one of the values (last write wins)
            assert 0.4 <= ctx.server.config["temperature"] <= 1.0

            # Clean up
            for client in clients:
                await client.close()

    @pytest.mark.asyncio
    @pytest.mark.timeout(20)  # Add timeout like successful thread test
    async def test_thread_local_resources(self):
        """Test thread-local resource management"""
        async with RealServerTestContext() as ctx:
            # Track thread IDs used for processing
            thread_ids = set()

            # Monkey-patch to track thread IDs
            original_process = ctx.server._process_text

            def tracking_process(*args, **kwargs):
                thread_ids.add(threading.current_thread().ident)
                return original_process(*args, **kwargs)

            ctx.server._process_text = tracking_process

            # Send multiple messages with reduced load (3 clients, 1 message each)
            results = await simulate_concurrent_clients(
                ctx.actual_port, num_clients=3, messages_per_client=1
            )

            # Should have used multiple threads
            assert len(thread_ids) > 1
            assert all(len(r) > 0 for r in results)


class TestMemoryManagement:
    """Test memory and resource management"""

    @pytest.mark.asyncio
    async def test_generator_cleanup(self):
        """Test that generators are properly cleaned up"""
        async with RealServerTestContext() as ctx:
            client = ctx.create_client()
            await client.connect()

            # Track generator count
            initial_count = len(ctx.server.client_generators)

            # Process message
            await client.send({"type": "text_input", "content": "Test message"})

            # Wait for completion
            await client.receive_all()

            # Close and wait for cleanup
            await client.close()
            await asyncio.sleep(0.5)

            # Generators should be cleaned up
            final_count = len(ctx.server.client_generators)
            assert final_count <= initial_count

    @pytest.mark.asyncio
    async def test_queue_cleanup(self):
        """Test that client queues are properly cleaned up"""
        async with RealServerTestContext() as ctx:
            # Track initial state
            initial_queues = len(ctx.server.client_queues)

            # Create and disconnect multiple clients
            for i in range(5):
                client = ctx.create_client()
                await client.connect()

                await client.send({"type": "text_input", "content": f"Message {i}"})

                # Don't wait for response, just disconnect
                await client.close()

            # Wait for cleanup
            await asyncio.sleep(1.0)

            # All queues should be cleaned up
            final_queues = len(ctx.server.client_queues)
            assert final_queues == initial_queues


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
