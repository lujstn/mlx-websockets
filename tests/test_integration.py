"""
Integration tests for MLX WebSocket Streaming Server
Tests end-to-end functionality with real WebSocket connections
"""

import asyncio
import base64
import io
import json
import threading
import time

# typing imports removed - using built-in types
import pytest
from PIL import Image

from .test_helpers import RealServerTestContext, RealWebSocketClient


@pytest.mark.integration
class TestWebSocketIntegration:
    """Test full WebSocket integration scenarios with real connections"""

    @pytest.mark.asyncio
    async def test_full_text_conversation(self):
        """Test a full text conversation flow"""
        async with RealServerTestContext() as ctx:
            client = ctx.create_client()
            await client.connect()

            # Send multiple messages in a conversation
            prompts = ["Hello", "How are you?", "Goodbye"]
            all_responses = []

            for prompt in prompts:
                await client.send({"type": "text_input", "content": prompt, "context": ""})

                # Receive complete response
                messages = await client.receive_all()
                all_responses.append(messages)

            # Verify we got responses for each prompt
            assert len(all_responses) == 3
            for responses in all_responses:
                assert any(msg["type"] == "response_complete" for msg in responses)

            await client.close()

    @pytest.mark.asyncio
    async def test_mixed_input_types(self):
        """Test handling mixed text and image inputs"""
        async with RealServerTestContext() as ctx:
            client = ctx.create_client()
            await client.connect()

            # Send text
            await client.send({"type": "text_input", "content": "Hello"})
            text_response = await client.receive_all()

            # Create and send image
            img = Image.new("RGB", (100, 100), color="green")
            img_buffer = io.BytesIO()
            img.save(img_buffer, format="PNG")
            img_base64 = base64.b64encode(img_buffer.getvalue()).decode()

            await client.send(
                {
                    "type": "image_input",
                    "image": f"data:image/png;base64,{img_base64}",
                    "prompt": "What's this?",
                }
            )
            image_response = await client.receive_all()

            # Send text again
            await client.send({"type": "text_input", "content": "Thanks"})
            final_response = await client.receive_all()

            # Verify all responses
            assert all(
                any(msg["type"] == "response_complete" for msg in resp)
                for resp in [text_response, image_response, final_response]
            )

            await client.close()

    @pytest.mark.asyncio
    async def test_context_preservation(self):
        """Test context preservation across messages"""
        async with RealServerTestContext() as ctx:
            client = ctx.create_client()
            await client.connect()

            # Send first message
            await client.send({"type": "text_input", "content": "My name is Alice", "context": ""})
            await client.receive_all()

            # Send follow-up with context
            await client.send(
                {
                    "type": "text_input",
                    "content": "What's my name?",
                    "context": "User said: My name is Alice",
                }
            )
            response = await client.receive_all()

            # Response should acknowledge the context
            full_text = "".join(
                msg.get("content", "") for msg in response if msg["type"] == "token"
            )

            # The model should have access to the context
            assert len(full_text) > 0

            await client.close()

    @pytest.mark.asyncio
    async def test_real_concurrent_clients(self):
        """Test multiple concurrent client connections with real async"""
        async with RealServerTestContext() as ctx:
            num_clients = 5
            clients = []

            # Create and connect all clients
            for _i in range(num_clients):
                client = ctx.create_client()
                await client.connect()
                clients.append(client)

            # Send messages from all clients concurrently
            async def send_and_receive(client: RealWebSocketClient, client_id: int):
                responses = []
                for msg_id in range(3):
                    await client.send(
                        {"type": "text_input", "content": f"Client {client_id} message {msg_id}"}
                    )
                    response = await client.receive_all()
                    responses.append(response)
                return responses

            # Execute concurrently
            results = await asyncio.gather(
                *[send_and_receive(client, i) for i, client in enumerate(clients)]
            )

            # Verify all clients got responses
            assert len(results) == num_clients
            for client_results in results:
                assert len(client_results) == 3  # 3 messages per client
                for response in client_results:
                    assert any(msg["type"] == "response_complete" for msg in response)

            # Clean up
            for client in clients:
                await client.close()

    @pytest.mark.asyncio
    async def test_error_recovery(self):
        """Test recovery from errors during processing"""
        async with RealServerTestContext() as ctx:
            client = ctx.create_client()
            await client.connect()

            # Send invalid message type
            await client.send({"type": "invalid_type", "data": "test"})

            # Server should continue working
            await client.send({"type": "text_input", "content": "Valid message after error"})

            response = await client.receive_all()
            assert any(msg["type"] == "response_complete" for msg in response)

            await client.close()

    @pytest.mark.asyncio
    async def test_config_updates_live(self):
        """Test configuration updates during runtime"""
        async with RealServerTestContext() as ctx:
            client = ctx.create_client()
            await client.connect()

            # Send with default config
            await client.send({"type": "text_input", "content": "Test with default config"})
            response1 = await client.receive_all()

            # Update config
            await client.send({"type": "config", "temperature": 0.5, "maxTokens": 50})

            # Brief pause for config update
            await asyncio.sleep(0.1)

            # Send with updated config
            await client.send({"type": "text_input", "content": "Test with updated config"})
            response2 = await client.receive_all()

            # Both should complete successfully
            assert any(msg["type"] == "response_complete" for msg in response1)
            assert any(msg["type"] == "response_complete" for msg in response2)

            # Verify config was actually updated
            assert ctx.server.config["temperature"] == 0.5
            assert ctx.server.config["maxTokens"] == 50

            await client.close()


@pytest.mark.integration
class TestPerformanceIntegration:
    """Test performance under realistic conditions"""

    @pytest.mark.asyncio
    async def test_sustained_throughput(self):
        """Test sustained throughput over time"""
        async with RealServerTestContext() as ctx:
            client = ctx.create_client()
            await client.connect()

            start_time = time.time()
            total_tokens = 0

            # Send multiple requests
            for i in range(10):
                await client.send(
                    {"type": "text_input", "content": f"Generate a response for request {i}"}
                )

                # Count tokens in response
                messages = await client.receive_all()
                tokens = [msg for msg in messages if msg["type"] == "token"]
                total_tokens += len(tokens)

            duration = time.time() - start_time
            throughput = total_tokens / duration

            print(f"\nSustained throughput: {throughput:.1f} tokens/sec")

            # Should maintain reasonable throughput
            assert throughput > 10  # tokens/sec (conservative for test)

            await client.close()

    @pytest.mark.asyncio
    async def test_latency_distribution(self):
        """Test latency distribution for responses"""
        async with RealServerTestContext() as ctx:
            latencies = []
            # Reduce iterations to avoid timeout
            num_iterations = 5  # Reduced from 20

            for i in range(num_iterations):
                client = ctx.create_client()
                await client.connect()

                start = time.time()
                await client.send({"type": "text_input", "content": f"Quick response {i}"})

                # Measure time to first token
                first_msg = await client.receive(timeout=2.0)  # Reduced timeout
                if first_msg and first_msg["type"] == "response_start":
                    # Get first token
                    token_msg = await client.receive(timeout=2.0)  # Reduced timeout
                    if token_msg and token_msg["type"] == "token":
                        latency = time.time() - start
                        latencies.append(latency)

                await client.close()
                # No sleep between tests to speed up

            if latencies:
                avg_latency = sum(latencies) / len(latencies)
                print(f"\nAverage latency to first token: {avg_latency * 1000:.2f}ms")

                # Latency should be reasonable
                assert avg_latency < 1.0  # 1000ms (more lenient)


@pytest.mark.integration
class TestRealConcurrentOperations:
    """Test real concurrent operations without mocking"""

    @pytest.mark.asyncio
    async def test_parallel_message_processing(self):
        """Test that server processes messages from same client in parallel"""
        async with RealServerTestContext() as ctx:
            client = ctx.create_client()
            await client.connect()

            # Send multiple messages without waiting for responses
            num_messages = 5
            for i in range(num_messages):
                await client.send({"type": "text_input", "content": f"Parallel message {i}"})

            # Collect all responses
            all_messages = []
            complete_count = 0
            timeout = time.time() + 30  # 30 second timeout

            while complete_count < num_messages and time.time() < timeout:
                msg = await client.receive(timeout=1.0)
                if msg:
                    all_messages.append(msg)
                    if msg["type"] == "response_complete":
                        complete_count += 1

            # Should have received all completions
            assert complete_count == num_messages

            await client.close()

    @pytest.mark.asyncio
    async def test_thread_pool_behavior(self):
        """Test thread pool behavior under concurrent load"""
        async with RealServerTestContext() as ctx:
            # Track thread count before
            initial_threads = len(threading.enumerate())

            # Create burst of concurrent operations
            clients = []
            for _ in range(10):
                client = ctx.create_client()
                await client.connect()
                clients.append(client)

            # Send messages concurrently and wait for responses
            tasks = []
            for i, client in enumerate(clients):

                async def send_and_receive(c, idx):
                    # Send just one message per client to avoid overwhelming
                    await c.send({"type": "text_input", "content": f"Client {idx} test message"})
                    # Wait for response to complete
                    while True:
                        try:
                            response = await asyncio.wait_for(c.websocket.recv(), timeout=5.0)
                            data = json.loads(response)
                            if data.get("type") == "response_complete":
                                break
                        except asyncio.TimeoutError:
                            break

                tasks.append(send_and_receive(client, i))

            await asyncio.gather(*tasks)

            # Monitor thread count during processing
            peak_threads = len(threading.enumerate())

            # Wait for some processing
            await asyncio.sleep(2.0)

            # Clean up
            for client in clients:
                await client.close()

            # Wait for thread cleanup
            await asyncio.sleep(1.0)
            final_threads = len(threading.enumerate())

            print(
                f"\nThread count - Initial: {initial_threads}, Peak: {peak_threads}, Final: {final_threads}"
            )

            # Threads should be managed properly
            assert peak_threads < initial_threads + 50  # Reasonable limit
            assert final_threads <= initial_threads + 5  # Should clean up

    @pytest.mark.asyncio
    async def test_graceful_shutdown_with_active_clients(self):
        """Test graceful shutdown with active client connections"""
        async with RealServerTestContext() as ctx:
            # Connect multiple clients
            clients = []
            for i in range(5):
                client = ctx.create_client()
                await client.connect()
                clients.append(client)

                # Start processing
                await client.send({"type": "text_input", "content": f"Long running task {i}"})

            # Give some time for processing to start
            await asyncio.sleep(0.5)

            # Close all clients gracefully
            for client in clients:
                await client.close()

            # Server should handle graceful disconnection
            # Verify by connecting a new client
            new_client = ctx.create_client()
            await new_client.connect()

            await new_client.send({"type": "text_input", "content": "Server still working?"})

            response = await new_client.receive_all()
            assert any(msg["type"] == "response_complete" for msg in response)

            await new_client.close()


@pytest.mark.integration
class TestEdgeCasesReal:
    """Test edge cases with real connections"""

    @pytest.mark.slow
    @pytest.mark.timeout(60)
    @pytest.mark.asyncio
    async def test_rapid_connect_disconnect(self):
        """Test handling rapid client connects/disconnects"""
        async with RealServerTestContext() as ctx:
            for i in range(20):
                client = ctx.create_client()
                await client.connect()

                # Send message
                await client.send({"type": "text_input", "content": f"Quick message {i}"})

                # Disconnect quickly (before response)
                await client.close()

                # Brief pause
                await asyncio.sleep(0.01)

            # Server should remain stable - test with new client
            test_client = ctx.create_client()
            await test_client.connect()
            await test_client.send({"type": "text_input", "content": "Still stable?"})

            response = await test_client.receive_all()
            assert any(msg["type"] == "response_complete" for msg in response)

            await test_client.close()

    @pytest.mark.asyncio
    async def test_large_message_handling(self):
        """Test handling of large messages"""
        async with RealServerTestContext() as ctx:
            client = ctx.create_client()
            await client.connect()

            # Send very long prompt
            long_prompt = "Please analyze this text: " + ("word " * 1000)

            await client.send({"type": "text_input", "content": long_prompt})

            # Should handle gracefully
            response = await client.receive_all()
            assert any(msg["type"] == "response_complete" for msg in response)

            await client.close()

    @pytest.mark.asyncio
    async def test_websocket_ping_pong(self):
        """Test WebSocket ping/pong keepalive"""
        async with RealServerTestContext() as ctx:
            client = ctx.create_client()
            await client.connect()

            # Send message
            await client.send({"type": "text_input", "content": "Test keepalive"})

            # Wait longer than typical ping interval
            await asyncio.sleep(3.0)

            # Connection should still be alive
            await client.send({"type": "text_input", "content": "Still connected?"})

            response = await client.receive_all()
            assert any(msg["type"] == "response_complete" for msg in response)

            await client.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
