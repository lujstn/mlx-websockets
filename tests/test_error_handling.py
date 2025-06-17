"""Error handling and edge case tests for MLX WebSocket Server."""

import asyncio
import json

import pytest
import websockets

from .test_helpers import RealServerTestContext, mock_mlx_models, mock_mlx_models_context


class TestErrorScenarios:
    """Test error handling and recovery scenarios"""

    @pytest.mark.asyncio
    async def test_malformed_json_handling(self):
        """Test handling of malformed JSON messages"""
        async with RealServerTestContext() as ctx:
            # Connect directly to WebSocket
            import websockets

            uri = f"ws://localhost:{ctx.actual_port}"

            async with websockets.connect(uri) as ws:
                # Send malformed JSON
                await ws.send("not valid json")

                # Send valid message after
                await ws.send(
                    json.dumps({"type": "text_input", "content": "Valid message after error"})
                )

                # Should still receive response for valid message
                response_received = False
                timeout = asyncio.create_task(asyncio.sleep(5.0))

                while not timeout.done():
                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=0.5)
                        data = json.loads(msg)
                        if data.get("type") == "response_complete":
                            response_received = True
                            break
                    except asyncio.TimeoutError:
                        continue

                assert response_received, "Server should recover from malformed JSON"

    @pytest.mark.asyncio
    async def test_invalid_image_data(self):
        """Test handling of invalid image data"""
        async with RealServerTestContext() as ctx:
            client = ctx.create_client()
            await client.connect()

            # Send invalid base64 image
            await client.send(
                {
                    "type": "image_input",
                    "image": "data:image/png;base64,invalid_base64_data",
                    "prompt": "What's this?",
                }
            )

            # Server should handle gracefully
            # Send valid message after
            await client.send({"type": "text_input", "content": "Valid text after invalid image"})

            response = await client.receive_all()
            assert any(msg["type"] == "response_complete" for msg in response)

            await client.close()

    @pytest.mark.asyncio
    async def test_missing_required_fields(self):
        """Test handling of messages with missing required fields"""
        async with RealServerTestContext() as ctx:
            client = ctx.create_client()
            await client.connect()

            # Send message without required 'content' field
            await client.send(
                {
                    "type": "text_input",
                    # Missing 'content'
                }
            )

            # Send valid message
            await client.send({"type": "text_input", "content": "Valid message with all fields"})

            response = await client.receive_all()
            assert any(msg["type"] == "response_complete" for msg in response)

            await client.close()

    @pytest.mark.asyncio
    async def test_concurrent_config_updates(self):
        """Test race conditions in concurrent config updates"""
        async with RealServerTestContext() as ctx:
            clients = []

            # Create multiple clients
            for _ in range(5):
                client = ctx.create_client()
                await client.connect()
                clients.append(client)

            # Send concurrent config updates
            async def update_config(client, temp):
                await client.send(
                    {
                        "type": "config",
                        "temperature": temp,
                        "maxTokens": 100 + int(temp * 100),
                    }
                )

            # Execute concurrent updates
            tasks = [update_config(client, 0.1 * i) for i, client in enumerate(clients)]
            await asyncio.gather(*tasks)

            # Wait for processing
            await asyncio.sleep(0.5)

            # Config should have valid values (last write wins)
            assert 0.0 <= ctx.server.config["temperature"] <= 1.0
            assert ctx.server.config["maxTokens"] >= 100

            # Clean up
            for client in clients:
                await client.close()

    @pytest.mark.asyncio
    async def test_client_queue_overflow(self):
        """Test handling when client message queue is full"""
        async with RealServerTestContext() as ctx:
            client = ctx.create_client()
            await client.connect()

            # Send many messages rapidly without reading responses
            for i in range(100):
                await client.send({"type": "text_input", "content": f"Rapid message {i}"})

            # Server should handle gracefully
            # Give time for processing
            await asyncio.sleep(2.0)

            # Should still be able to process new messages
            await client.send({"type": "text_input", "content": "Message after queue stress"})

            # Try to receive some responses
            received_any = False
            for _ in range(10):
                msg = await client.receive(timeout=1.0)
                if msg:
                    received_any = True
                    break

            assert received_any, "Server should still be responsive"

            await client.close()

    @pytest.mark.asyncio
    async def test_model_generation_error(self):
        """Test handling of errors during model generation"""
        # Use custom mock that raises exception
        with mock_mlx_models_context() as mocks:
            # Make generate raise exception after a few tokens
            token_count = 0

            def failing_generator(*args, **kwargs):
                nonlocal token_count
                for token in ["Hello", " ", "world"]:
                    token_count += 1
                    if token_count > 2:
                        raise RuntimeError("Model generation failed")
                    yield token

            mocks["generate"].side_effect = failing_generator

            async with RealServerTestContext() as ctx:
                client = ctx.create_client()
                await client.connect()

                # Send message that will trigger error
                await client.send(
                    {"type": "text_input", "content": "This will fail during generation"}
                )

                # Collect responses
                messages = await client.receive_all(timeout=5.0)

                # Should have received some tokens before error
                tokens = [msg for msg in messages if msg["type"] == "token"]
                assert len(tokens) >= 2

                # Server should still be functional
                await client.send({"type": "text_input", "content": "Message after error"})

                # Should get response
                response = await client.receive_all()
                assert len(response) > 0

                await client.close()

    @pytest.mark.asyncio
    async def test_websocket_close_during_generation(self):
        """Test client disconnect during active generation"""
        async with RealServerTestContext() as ctx:
            client = ctx.create_client()
            await client.connect()

            # Send message to start generation
            await client.send({"type": "text_input", "content": "Start long generation process"})

            # Receive a few tokens
            for _ in range(3):
                msg = await client.receive(timeout=1.0)
                if msg and msg["type"] == "token":
                    break

            # Close connection abruptly
            await client.websocket.close()

            # Server should clean up properly
            await asyncio.sleep(1.0)

            # New client should work fine
            new_client = ctx.create_client()
            await new_client.connect()

            await new_client.send({"type": "text_input", "content": "New client after disconnect"})

            response = await new_client.receive_all()
            assert any(msg["type"] == "response_complete" for msg in response)

            await new_client.close()

    @pytest.mark.asyncio
    async def test_invalid_config_values(self):
        """Test handling of invalid configuration values"""
        async with RealServerTestContext() as ctx:
            client = ctx.create_client()
            await client.connect()

            # Send invalid config values
            test_cases = [
                {"temperature": -1.0},  # Negative temperature
                {"temperature": 2.0},  # Too high temperature
                {"maxOutputTokens": -10},  # Negative tokens
                {"maxOutputTokens": "not a number"},  # Wrong type
                {"topP": 1.5},  # Invalid probability
                {"unknownParam": "value"},  # Unknown parameter
            ]

            for config in test_cases:
                await client.send({"type": "config", **config})

            # Server should handle all gracefully
            await asyncio.sleep(0.5)

            # Should still work with valid message
            await client.send({"type": "text_input", "content": "Message after invalid configs"})

            response = await client.receive_all()
            assert any(msg["type"] == "response_complete" for msg in response)

            await client.close()

    @pytest.mark.asyncio
    async def test_very_large_prompt(self):
        """Test handling of very large prompts"""
        async with RealServerTestContext() as ctx:
            client = ctx.create_client()
            await client.connect()

            # Test 1: A prompt that's under WebSocket's default 1MB limit but still large
            large_prompt = "x" * (500 * 1024)  # 500KB - should work fine

            await client.send({"type": "text_input", "content": large_prompt})

            # Server should handle this normally
            response = await client.receive_all(timeout=5.0)
            assert any(msg["type"] == "response_complete" for msg in response)

            # Test 2: Try to send something that will exceed WebSocket frame limit
            # This tests that the server survives protocol-level errors
            await client.close()
            client = ctx.create_client()
            await client.connect()

            # Try to send a message that exceeds default WebSocket limits
            # This will fail at the protocol level
            huge_prompt = "x" * (2 * 1024 * 1024)  # 2MB
            try:
                await client.send({"type": "text_input", "content": huge_prompt})
                # If send succeeds, try to receive to trigger the error
                await client.receive(timeout=1.0)
            except (websockets.exceptions.ConnectionClosedError, ConnectionError):
                # Expected - WebSocket protocol error
                pass

            # Test 3: Verify server is still healthy after protocol error
            client = ctx.create_client()
            await client.connect()

            # Server should still be responsive with normal messages
            await client.send(
                {"type": "text_input", "content": "Small message after protocol error"}
            )

            response = await client.receive_all()
            assert any(msg["type"] == "response_complete" for msg in response)

            await client.close()
