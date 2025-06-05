"""Performance benchmarks for MLX WebSocket Server."""

import asyncio
import json
import statistics
import time

import pytest

from .test_helpers import ServerTestContext, mock_generate_streaming


@pytest.mark.benchmark
class TestPerformanceBenchmarks:
    """Performance benchmark tests"""

    def test_token_generation_throughput(self):
        """Measure token generation throughput"""
        with ServerTestContext() as ctx:
            # Fast token generation
            tokens = ["Token"] * 1000  # 1000 tokens

            def mock_generator(*args, **kwargs):
                yield from tokens

            ctx.mocks["generate"].side_effect = mock_generator

            # Create websocket and collect messages
            messages = []
            websocket = ctx.create_websocket(messages)

            # Process text
            start_time = time.time()
            ctx.process_text(websocket, "Generate many tokens")
            end_time = time.time()

            # Wait for async operations
            time.sleep(0.5)

            # Extract tokens
            token_count = sum(1 for msg in messages if json.loads(msg).get("type") == "token")
            total_time = end_time - start_time
            throughput = token_count / total_time if total_time > 0 else 0

            print("\nToken Generation Metrics:")
            print(f"  Total tokens: {token_count}")
            print(f"  Total time: {total_time:.2f}s")
            print(f"  Throughput: {throughput:.0f} tokens/sec")

            # Verify high throughput
            assert token_count == 1000, f"Expected 1000 tokens, got {token_count}"
            assert throughput > 100, f"Expected >100 tokens/sec, got {throughput:.0f}"

    def test_concurrent_client_handling(self):
        """Test handling multiple concurrent clients"""
        with ServerTestContext() as ctx:
            # Mock generate to return different responses per client
            def mock_generate(*args, **kwargs):
                prompt = kwargs.get("prompt", "")
                if "client1" in prompt:
                    for token in ["Response", " ", "for", " ", "client", " ", "1"]:
                        yield token
                elif "client2" in prompt:
                    for token in ["Response", " ", "for", " ", "client", " ", "2"]:
                        yield token
                else:
                    yield "Default response"

            ctx.mocks["generate"].side_effect = mock_generate

            # Create websockets for two clients
            messages1 = []
            messages2 = []
            ws1 = ctx.create_websocket(messages1)
            ws2 = ctx.create_websocket(messages2)

            # Process both clients concurrently
            import threading

            def process_client1():
                ctx.process_text(ws1, "Hello from client1", ("127.0.0.1", 12345))

            def process_client2():
                ctx.process_text(ws2, "Hello from client2", ("127.0.0.1", 12346))

            thread1 = threading.Thread(target=process_client1)
            thread2 = threading.Thread(target=process_client2)

            thread1.start()
            thread2.start()

            thread1.join(timeout=5.0)
            thread2.join(timeout=5.0)

            # Wait for async operations
            time.sleep(0.5)

            # Verify both clients got responses
            assert len(messages1) > 0, "Client 1 should receive messages"
            assert len(messages2) > 0, "Client 2 should receive messages"

            # Check responses are correct
            response1 = None
            response2 = None

            for msg in messages1:
                data = json.loads(msg)
                if data.get("type") == "response_complete":
                    response1 = data.get("full_text")

            for msg in messages2:
                data = json.loads(msg)
                if data.get("type") == "response_complete":
                    response2 = data.get("full_text")

            assert response1 == "Response for client 1"
            assert response2 == "Response for client 2"

            print("\nConcurrent client test passed!")
            print(f"  Client 1 messages: {len(messages1)}")
            print(f"  Client 2 messages: {len(messages2)}")

    def test_memory_efficiency(self):
        """Test memory usage doesn't grow excessively"""
        with ServerTestContext() as ctx:
            import gc

            import mlx.core as mx

            # Mock generate
            ctx.mocks["generate"].side_effect = mock_generate_streaming("Memory test response")

            # Get initial memory
            gc.collect()
            initial_memory = mx.metal.get_active_memory() / 1024 / 1024  # MB

            # Process multiple requests
            messages = []
            websocket = ctx.create_websocket(messages)

            for i in range(10):
                ctx.process_text(websocket, f"Request {i}")
                time.sleep(0.1)

            # Get final memory
            gc.collect()
            final_memory = mx.metal.get_active_memory() / 1024 / 1024  # MB

            memory_increase = final_memory - initial_memory

            print("\nMemory Usage:")
            print(f"  Initial: {initial_memory:.2f} MB")
            print(f"  Final: {final_memory:.2f} MB")
            print(f"  Increase: {memory_increase:.2f} MB")

            # Memory increase should be minimal
            assert memory_increase < 50, f"Memory increased by {memory_increase:.2f} MB"

    def test_large_context_handling(self):
        """Test handling large context windows"""
        with ServerTestContext() as ctx:
            # Generate response with context consideration
            def mock_generate_with_context(*args, **kwargs):
                prompt = kwargs.get("prompt", "")
                # Simulate processing large context
                time.sleep(0.1)  # Simulate processing time
                yield f"Processed {len(prompt)} characters"

            ctx.mocks["generate"].side_effect = mock_generate_with_context

            # Create large context
            large_context = "Background information. " * 500  # ~10KB of context

            messages = []
            websocket = ctx.create_websocket(messages)

            start_time = time.time()
            ctx.process_text(websocket, "Summarize this", context=large_context)
            end_time = time.time()

            # Wait for completion
            time.sleep(0.5)

            # Verify response
            response = None
            for msg in messages:
                data = json.loads(msg)
                if data.get("type") == "response_complete":
                    response = data.get("full_text")

            assert response is not None, "Should receive response"
            assert "Processed" in response

            processing_time = end_time - start_time
            print("\nLarge context handling:")
            print(f"  Context size: {len(large_context)} chars")
            print(f"  Processing time: {processing_time:.2f}s")

    def test_error_recovery_performance(self):
        """Test performance of error handling and recovery"""
        with ServerTestContext() as ctx:
            # Mock generate to fail occasionally
            call_count = [0]

            def mock_generate_with_errors(*args, **kwargs):
                call_count[0] += 1
                if call_count[0] % 3 == 0:
                    raise Exception("Simulated error")
                yield f"Response {call_count[0]}"

            ctx.mocks["generate"].side_effect = mock_generate_with_errors

            messages = []
            websocket = ctx.create_websocket(messages)

            # Send multiple requests
            start_time = time.time()
            for i in range(10):
                ctx.process_text(websocket, f"Request {i}")
                time.sleep(0.1)
            end_time = time.time()

            # Count successful vs error responses
            success_count = 0
            error_count = 0

            for msg in messages:
                data = json.loads(msg)
                if data.get("type") == "response_complete":
                    success_count += 1
                elif data.get("type") == "error":
                    error_count += 1

            total_time = end_time - start_time
            avg_time_per_request = total_time / 10

            print("\nError recovery performance:")
            print("  Total requests: 10")
            print(f"  Successful: {success_count}")
            print(f"  Errors: {error_count}")
            print(f"  Total time: {total_time:.2f}s")
            print(f"  Avg time per request: {avg_time_per_request:.2f}s")

            # Should handle errors gracefully
            assert success_count > 0, "Should have some successful responses"
            assert error_count > 0, "Should have some error responses"
            assert success_count + error_count >= 10, "Should process all requests"
