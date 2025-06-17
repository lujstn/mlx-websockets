"""Performance and benchmark tests for MLX WebSocket Server."""

import asyncio
import time

import pytest

from .test_helpers import RealServerTestContext


class TestPerformance:
    """Performance benchmark tests using real server"""

    @pytest.mark.asyncio
    async def test_token_generation_throughput(self):
        """Measure token generation throughput"""
        async with RealServerTestContext() as ctx:
            client = ctx.create_client()
            await client.connect()

            # Track tokens received
            total_tokens = 0
            start_time = time.time()

            # Send multiple messages
            for i in range(5):
                await client.send(
                    {"type": "text_input", "content": f"Generate tokens for message {i}"}
                )

                # Count tokens
                messages = await client.receive_all()
                tokens = [msg for msg in messages if msg["type"] == "token"]
                total_tokens += len(tokens)

            duration = time.time() - start_time
            throughput = total_tokens / duration

            print(f"\nToken throughput: {throughput:.1f} tokens/sec")
            assert throughput > 5  # Minimum expected throughput

            await client.close()

    @pytest.mark.asyncio
    async def test_concurrent_client_scalability(self):
        """Test server scalability with concurrent clients"""
        async with RealServerTestContext() as ctx:
            num_clients = 10
            messages_per_client = 3

            # Track timing
            start_time = time.time()

            # Create clients
            clients = []
            for _ in range(num_clients):
                client = ctx.create_client()
                await client.connect()
                clients.append(client)

            # Send messages concurrently
            async def process_client(client, client_id):
                responses = []
                for msg_id in range(messages_per_client):
                    await client.send(
                        {"type": "text_input", "content": f"Client {client_id} msg {msg_id}"}
                    )
                    resp = await client.receive_all()
                    responses.append(resp)
                return len(responses)

            # Execute all clients concurrently
            results = await asyncio.gather(
                *[process_client(client, i) for i, client in enumerate(clients)]
            )

            # Clean up
            for client in clients:
                await client.close()

            duration = time.time() - start_time
            total_messages = sum(results)

            print(
                f"\nProcessed {total_messages} messages from {num_clients} clients in {duration:.2f}s"
            )
            print(f"Average time per message: {duration / total_messages * 1000:.2f}ms")

            # All clients should have completed
            assert all(r == messages_per_client for r in results)
            assert duration < 30  # Should complete within 30 seconds

    @pytest.mark.asyncio
    async def test_memory_stability(self):
        """Test memory stability under sustained load"""
        async with RealServerTestContext() as ctx:
            client = ctx.create_client()
            await client.connect()

            # Send many messages to test memory handling
            for i in range(20):
                await client.send({"type": "text_input", "content": f"Memory test message {i}"})

                # Receive and discard
                await client.receive_all()

                # Brief pause
                await asyncio.sleep(0.1)

            # Server should still be responsive
            await client.send(
                {"type": "text_input", "content": "Final message after sustained load"}
            )

            final_response = await client.receive_all()
            assert any(msg["type"] == "response_complete" for msg in final_response)

            await client.close()

    @pytest.mark.asyncio
    async def test_response_time_consistency(self):
        """Test response time consistency"""
        async with RealServerTestContext() as ctx:
            response_times = []

            for i in range(10):
                client = ctx.create_client()
                await client.connect()

                start = time.time()
                await client.send({"type": "text_input", "content": f"Timing test {i}"})

                # Wait for first token
                first_msg = await client.receive(timeout=5.0)
                if first_msg and first_msg["type"] == "response_start":
                    token_msg = await client.receive(timeout=5.0)
                    if token_msg and token_msg["type"] == "token":
                        response_time = time.time() - start
                        response_times.append(response_time)

                await client.close()
                await asyncio.sleep(0.1)

            if response_times:
                avg_time = sum(response_times) / len(response_times)
                max_time = max(response_times)
                min_time = min(response_times)

                print(
                    f"\nResponse times - Avg: {avg_time * 1000:.2f}ms, "
                    f"Min: {min_time * 1000:.2f}ms, Max: {max_time * 1000:.2f}ms"
                )

                # Response times should be consistent
                # Allow 5x variance to handle system load and first connection overhead
                assert max_time < 5 * avg_time  # Max should not be more than 5x average
                assert avg_time < 1.0  # Average should be under 1 second
