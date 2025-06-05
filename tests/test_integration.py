"""
Integration tests for MLX WebSocket Streaming Server
Tests end-to-end functionality and WebSocket communication
"""

import asyncio
import base64
import io
import json
import threading
import time

import pytest
from PIL import Image

from .test_helpers import ServerTestContext


@pytest.mark.integration
class TestWebSocketIntegration:
    """Test full WebSocket integration scenarios"""

    def test_full_text_conversation(self):
        """Test a full text conversation flow"""
        with ServerTestContext() as ctx:
            # Different responses for different prompts
            responses = {
                "Hello": ["Hi", " ", "there", "!"],
                "How are you?": ["I'm", " ", "doing", " ", "great", "!"],
                "Goodbye": ["See", " ", "you", " ", "later", "!"],
            }

            def get_response(*args, **kwargs):
                prompt = kwargs.get("prompt", "")
                for key, response in responses.items():
                    if key in prompt:
                        return iter(response)
                return iter(["Default", " ", "response"])

            ctx.mocks["generate"].side_effect = get_response

            # Simulate full conversation
            messages = []
            websocket = ctx.create_websocket(messages)

            # Send multiple messages
            ctx.process_text(websocket, "Hello")
            time.sleep(0.5)

            ctx.process_text(websocket, "How are you?")
            time.sleep(0.5)

            ctx.process_text(websocket, "Goodbye")
            time.sleep(0.5)

            # Parse all messages
            parsed = [json.loads(msg) for msg in messages]

            # Verify conversation flow
            responses_complete = [msg for msg in parsed if msg["type"] == "response_complete"]

            assert len(responses_complete) >= 3
            assert "Hi there!" in responses_complete[0]["full_text"]
            assert "I'm doing great!" in responses_complete[1]["full_text"]
            assert "See you later!" in responses_complete[2]["full_text"]

    def test_mixed_input_types(self):
        """Test handling mixed text and image inputs"""
        with ServerTestContext() as ctx:

            def get_response(*args, **kwargs):
                if "image" in kwargs:
                    return iter(["Image", " ", "description"])
                else:
                    return iter(["Text", " ", "response"])

            ctx.mocks["generate"].side_effect = get_response

            # Create test image
            img = Image.new("RGB", (100, 100), color="green")
            img_buffer = io.BytesIO()
            img.save(img_buffer, format="PNG")
            img_base64 = base64.b64encode(img_buffer.getvalue()).decode()

            messages = []
            websocket = ctx.create_websocket(messages)

            # Send text
            ctx.process_text(websocket, "Hello")
            time.sleep(0.5)

            # Send image
            import threading

            ctx.server._process_image(
                {
                    "type": "image",
                    "timestamp": time.time(),
                    "content": f"data:image/png;base64,{img_base64}",
                    "prompt": "What's this?",
                    "source": "test",
                },
                websocket,
                ctx.loop,
                ("127.0.0.1", 12345),
                threading.Event(),
            )
            time.sleep(0.5)

            # Send text again
            ctx.process_text(websocket, "Thanks")
            time.sleep(0.5)

            # Parse responses
            parsed = [json.loads(msg) for msg in messages]
            responses_complete = [msg for msg in parsed if msg["type"] == "response_complete"]

            # Verify mixed responses
            assert len(responses_complete) >= 3
            assert "Text response" in responses_complete[0]["full_text"]
            assert "Image description" in responses_complete[1]["full_text"]
            assert "Text response" in responses_complete[2]["full_text"]

    def test_context_preservation(self):
        """Test context preservation across messages"""
        with ServerTestContext() as ctx:
            captured_prompts = []

            def capture_prompts(*args, **kwargs):
                captured_prompts.append(kwargs.get("prompt", ""))
                return iter(["Response"])

            ctx.mocks["generate"].side_effect = capture_prompts

            messages = []
            websocket = ctx.create_websocket(messages)

            # Send messages with increasing context
            ctx.process_text(websocket, "My name is Alice", context="")
            time.sleep(0.5)

            ctx.process_text(websocket, "What's my name?", context="User said: My name is Alice")
            time.sleep(0.5)

            # Verify context was included
            assert len(captured_prompts) >= 2
            assert "My name is Alice" in captured_prompts[0]
            assert "Context: User said: My name is Alice" in captured_prompts[1]
            assert "What's my name?" in captured_prompts[1]

    def test_concurrent_clients(self):
        """Test multiple concurrent client connections"""
        with ServerTestContext() as ctx:
            # Track which client sent each request

            def client_specific_response(*args, **kwargs):
                prompt = kwargs.get("prompt", "")
                if "Client1" in prompt:
                    return iter(["Response", " ", "for", " ", "Client1"])
                elif "Client2" in prompt:
                    return iter(["Response", " ", "for", " ", "Client2"])
                else:
                    return iter(["Unknown", " ", "client"])

            ctx.mocks["generate"].side_effect = client_specific_response

            # Create two clients
            messages1 = []
            messages2 = []
            ws1 = ctx.create_websocket(messages1)
            ws2 = ctx.create_websocket(messages2)

            # Process requests concurrently
            import threading

            def client1_work():
                for i in range(3):
                    ctx.process_text(ws1, f"Client1 message {i}", ("127.0.0.1", 12345))
                    time.sleep(0.1)

            def client2_work():
                for i in range(3):
                    ctx.process_text(ws2, f"Client2 message {i}", ("127.0.0.1", 12346))
                    time.sleep(0.1)

            thread1 = threading.Thread(target=client1_work)
            thread2 = threading.Thread(target=client2_work)

            thread1.start()
            thread2.start()

            thread1.join()
            thread2.join()

            # Wait for all async operations
            time.sleep(1.0)

            # Verify each client got correct responses
            parsed1 = [json.loads(msg) for msg in messages1]
            parsed2 = [json.loads(msg) for msg in messages2]

            responses1 = [msg for msg in parsed1 if msg["type"] == "response_complete"]
            responses2 = [msg for msg in parsed2 if msg["type"] == "response_complete"]

            assert all("Response for Client1" in r["full_text"] for r in responses1)
            assert all("Response for Client2" in r["full_text"] for r in responses2)

    def test_error_recovery(self):
        """Test recovery from errors during processing"""
        with ServerTestContext() as ctx:
            error_count = [0]

            def sometimes_fail(*args, **kwargs):
                error_count[0] += 1
                if error_count[0] % 2 == 0:
                    raise Exception("Simulated error")
                return iter(["Success", " ", "response"])

            ctx.mocks["generate"].side_effect = sometimes_fail

            messages = []
            websocket = ctx.create_websocket(messages)

            # Send multiple messages
            for i in range(5):
                ctx.process_text(websocket, f"Message {i}")
                time.sleep(0.1)

            # Wait for processing
            time.sleep(0.5)

            # Parse responses
            parsed = [json.loads(msg) for msg in messages]

            success_responses = [msg for msg in parsed if msg["type"] == "response_complete"]
            error_responses = [msg for msg in parsed if msg["type"] == "error"]

            # Should have mix of successes and errors
            assert len(success_responses) > 0
            assert len(error_responses) > 0
            # The server tries vision API first, then falls back to text API
            assert (
                "Text-only model API not available and vision API failed"
                in error_responses[0]["error"]
            )

    def test_config_updates(self):
        """Test configuration updates during runtime"""
        with ServerTestContext() as ctx:
            captured_configs = []

            def capture_config(*args, **kwargs):
                captured_configs.append(
                    {
                        "temperature": kwargs.get("temperature"),
                        "max_tokens": kwargs.get("max_tokens"),
                    }
                )
                return iter(["Config", " ", "test"])

            ctx.mocks["generate"].side_effect = capture_config

            messages = []
            websocket = ctx.create_websocket(messages)

            # Send with default config
            ctx.process_text(websocket, "Test 1")
            time.sleep(0.5)

            # Update config
            with ctx.server.config_lock:
                ctx.server.config["temperature"] = 0.5
                ctx.server.config["maxOutputTokens"] = 100

            # Send with updated config
            ctx.process_text(websocket, "Test 2")
            time.sleep(0.5)

            # Verify configs were captured
            assert len(captured_configs) >= 2
            assert captured_configs[0]["temperature"] != captured_configs[1]["temperature"]
            assert captured_configs[0]["max_tokens"] != captured_configs[1]["max_tokens"]


@pytest.mark.integration
class TestPerformanceIntegration:
    """Test performance under realistic conditions"""

    def test_sustained_throughput(self):
        """Test sustained throughput over time"""
        with ServerTestContext() as ctx:
            # Generate tokens at consistent rate
            def steady_generator(*args, **kwargs):
                for i in range(100):
                    yield f"Token{i} "

            ctx.mocks["generate"].side_effect = steady_generator

            messages = []
            websocket = ctx.create_websocket(messages)

            # Measure throughput over multiple requests
            start_time = time.time()

            for i in range(10):
                ctx.process_text(websocket, f"Request {i}")
                time.sleep(0.1)

            end_time = time.time()

            # Wait for all processing
            time.sleep(1.0)

            # Count total tokens
            parsed = [json.loads(msg) for msg in messages]
            total_tokens = sum(1 for msg in parsed if msg["type"] == "token")

            duration = end_time - start_time
            throughput = total_tokens / duration

            print("\nSustained throughput test:")
            print(f"  Total tokens: {total_tokens}")
            print(f"  Duration: {duration:.2f}s")
            print(f"  Throughput: {throughput:.0f} tokens/sec")

            # Should maintain good throughput
            assert throughput > 50  # tokens/sec

    def test_memory_stability(self):
        """Test memory stability over many requests"""
        with ServerTestContext() as ctx:
            import gc

            import mlx.core as mx

            # Simple generator
            ctx.mocks["generate"].side_effect = lambda *args, **kwargs: iter(
                ["Test", " ", "response"]
            )

            messages = []
            websocket = ctx.create_websocket(messages)

            # Get initial memory
            gc.collect()
            initial_memory = mx.metal.get_active_memory() / 1024 / 1024  # MB

            # Process many requests
            for i in range(100):
                ctx.process_text(websocket, f"Request {i}")
                if i % 10 == 0:
                    time.sleep(0.1)  # Brief pause every 10 requests

            # Wait for completion
            time.sleep(2.0)

            # Get final memory
            gc.collect()
            final_memory = mx.metal.get_active_memory() / 1024 / 1024  # MB

            memory_growth = final_memory - initial_memory

            print("\nMemory stability test:")
            print(f"  Initial memory: {initial_memory:.2f} MB")
            print(f"  Final memory: {final_memory:.2f} MB")
            print(f"  Memory growth: {memory_growth:.2f} MB")

            # Memory growth should be reasonable
            assert memory_growth < 100  # MB

    def test_latency_distribution(self):
        """Test latency distribution for responses"""
        with ServerTestContext() as ctx:
            # Fast generator
            ctx.mocks["generate"].side_effect = lambda *args, **kwargs: iter(["Quick"])

            latencies = []

            for i in range(50):
                messages = []
                websocket = ctx.create_websocket(messages)

                start = time.time()
                ctx.process_text(websocket, f"Request {i}")

                # Wait for response_start
                while not messages or "response_start" not in messages[0]:
                    time.sleep(0.001)

                latency = time.time() - start
                latencies.append(latency)

                time.sleep(0.05)  # Brief pause between requests

            # Calculate statistics
            import statistics

            avg_latency = statistics.mean(latencies)
            p50_latency = statistics.median(latencies)
            p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]
            p99_latency = sorted(latencies)[int(len(latencies) * 0.99)]

            print("\nLatency distribution test:")
            print(f"  Average: {avg_latency*1000:.2f}ms")
            print(f"  P50: {p50_latency*1000:.2f}ms")
            print(f"  P95: {p95_latency*1000:.2f}ms")
            print(f"  P99: {p99_latency*1000:.2f}ms")

            # Latency should be reasonable
            assert p95_latency < 0.1  # 100ms


@pytest.mark.integration
class TestEdgeCases:
    """Test edge cases and error conditions"""

    def test_rapid_client_disconnect(self):
        """Test handling rapid client connects/disconnects"""
        with ServerTestContext() as ctx:
            ctx.mocks["generate"].side_effect = lambda *args, **kwargs: iter(["Response"])

            # Simulate rapid connects/disconnects
            for i in range(20):
                messages = []
                websocket = ctx.create_websocket(messages)

                # Start processing
                stop_event = ctx.process_text(websocket, f"Message {i}")

                # Quickly disconnect
                time.sleep(0.01)
                stop_event.set()

                # Brief pause
                time.sleep(0.01)

            # Server should remain stable
            assert True  # If we get here, server didn't crash

    def test_malformed_requests(self):
        """Test handling of malformed requests"""
        with ServerTestContext() as ctx:
            # Test various malformed inputs via direct server methods
            messages = []
            websocket = ctx.create_websocket(messages)

            # Test with missing fields
            try:
                ctx.server._process_text(
                    {"type": "text"},  # Missing required fields
                    websocket,
                    ctx.loop,
                    ("127.0.0.1", 12345),
                    threading.Event(),
                )
            except KeyError:
                pass  # Expected

            # Test with invalid image data
            ctx.server._process_image(
                {
                    "type": "image",
                    "timestamp": time.time(),
                    "content": "not-base64-data",
                    "prompt": "Test",
                    "source": "test",
                },
                websocket,
                ctx.loop,
                ("127.0.0.1", 12345),
                threading.Event(),
            )

            time.sleep(0.5)

            # Should have error messages
            parsed = [json.loads(msg) for msg in messages]
            errors = [msg for msg in parsed if msg["type"] == "error"]
            assert len(errors) > 0

    def test_extreme_load_conditions(self):
        """Test behavior under extreme load"""
        with ServerTestContext() as ctx:
            # Very fast token generation
            def fast_generator():
                for i in range(10):
                    yield f"T{i}"

            ctx.mocks["generate"].side_effect = lambda *args, **kwargs: fast_generator()

            # Create multiple clients sending rapid requests
            clients = []
            for _ in range(10):
                messages = []
                ws = ctx.create_websocket(messages)
                clients.append((ws, messages))

            # Send many requests from each client
            import threading

            def flood_client(ws, client_id):
                for j in range(20):
                    ctx.process_text(
                        ws, f"Client{client_id} msg{j}", ("127.0.0.1", 12345 + client_id)
                    )
                    time.sleep(0.001)  # Minimal delay

            threads = []
            for i, (ws, _) in enumerate(clients):
                thread = threading.Thread(target=flood_client, args=(ws, i))
                threads.append(thread)
                thread.start()

            # Wait for all threads
            for thread in threads:
                thread.join()

            # Wait for processing
            time.sleep(2.0)

            # Verify some messages were processed from each client
            for _, messages in clients:
                parsed = [json.loads(msg) for msg in messages]
                completions = [msg for msg in parsed if msg["type"] == "response_complete"]
                assert len(completions) > 0  # Each client should get some responses


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
