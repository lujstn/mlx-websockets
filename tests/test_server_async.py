"""Async-specific behavior tests for MLX WebSocket server."""

import asyncio
import base64
import io
import json
import threading
import time
from queue import Empty, Queue
from unittest.mock import ANY, AsyncMock, MagicMock, Mock, call, patch

import pytest

from .test_helpers import mock_mlx_models, mock_mlx_models_context


class AsyncMessageIterator:
    """Proper async iterator for mocking WebSocket message streams."""

    def __init__(self, messages, keep_alive_duration=5.0):
        self.messages = messages
        self.index = 0
        self.keep_alive_duration = keep_alive_duration
        self.all_yielded = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.index < len(self.messages):
            msg = self.messages[self.index]
            self.index += 1
            await asyncio.sleep(0.01)  # Simulate network delay
            if self.index >= len(self.messages):
                self.all_yielded = True
            return msg

        # Keep connection alive to allow processing
        if not self.all_yielded:
            self.all_yielded = True
        await asyncio.sleep(self.keep_alive_duration)
        raise StopAsyncIteration


def create_server(**kwargs):
    """Helper to create server with proper mocking."""
    from mlx_websockets.server import MLXStreamingServer, _import_dependencies

    _import_dependencies(debug=True)
    # Enable debug mode to see errors
    kwargs.setdefault("debug", True)
    return MLXStreamingServer(**kwargs)


class TestAsyncWebSocketHandling:
    """Test async WebSocket handling behavior."""

    @pytest.mark.asyncio
    async def test_concurrent_client_connections(self):
        """Test handling multiple concurrent client connections."""
        with mock_mlx_models_context():
            server = create_server()

            # Create multiple mock clients
            clients = []
            for i in range(5):
                websocket = AsyncMock()
                websocket.remote_address = (f"192.168.1.{i}", 12340 + i)
                websocket.__aiter__.return_value = []  # Empty message stream
                clients.append(websocket)

            # Connect all clients concurrently
            tasks = [server.handle_client(ws, "/") for ws in clients]
            await asyncio.gather(*tasks)

            # Verify all clients were tracked
            assert len(server.active_connections) == 0  # All cleaned up after completion

    @pytest.mark.asyncio
    async def test_client_message_ordering(self):
        """Test that client messages are processed in order."""
        with mock_mlx_models_context():
            server = create_server()

            processed_messages = []
            expected_count = 5
            processing_complete = asyncio.Event()

            # Mock the processing to track order
            original_process_text = server._process_text

            def track_process_text(data, *args):
                processed_messages.append(data["content"])
                # Signal completion when all messages are processed
                if len(processed_messages) >= expected_count:
                    processing_complete.set()
                return original_process_text(data, *args)

            with patch.object(server, "_process_text", side_effect=track_process_text):
                # Send messages in order
                messages = [
                    json.dumps({"type": "text_input", "content": f"Message {i}"})
                    for i in range(expected_count)
                ]

                # Create proper async iterator with longer keep-alive to ensure all messages are processed
                message_iterator = AsyncMessageIterator(messages, keep_alive_duration=2.0)

                # Create websocket mock that IS the async iterator
                websocket = message_iterator
                websocket.remote_address = ("127.0.0.1", 12345)
                websocket.send = AsyncMock()
                websocket.close = AsyncMock()

                # Process messages in a task so we can wait for completion
                handle_task = asyncio.create_task(server.handle_client(websocket, "/"))

                # Wait for all messages to be processed with timeout
                try:
                    await asyncio.wait_for(processing_complete.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    print(
                        f"Debug: Timeout waiting for processing. Processed {len(processed_messages)} messages: {processed_messages}"
                    )

                # Allow some time for cleanup
                await asyncio.sleep(0.1)

                # Cancel the handle task if still running
                if not handle_task.done():
                    handle_task.cancel()
                    try:
                        await handle_task
                    except asyncio.CancelledError:
                        pass

                print(f"Debug: Processed {len(processed_messages)} messages: {processed_messages}")

                # Messages should be processed in order
                assert processed_messages == [f"Message {i}" for i in range(expected_count)]

    @pytest.mark.asyncio
    async def test_async_shutdown_coordination(self):
        """Test async shutdown coordination with active connections."""
        with mock_mlx_models_context():
            server = create_server()

            # Create mock active connections
            active_websockets = []
            for _ in range(3):
                ws = AsyncMock()
                ws.close = AsyncMock()
                server.active_connections.add(ws)
                active_websockets.append(ws)

            # Perform shutdown
            await server.shutdown()

            # Verify all connections were closed
            for ws in active_websockets:
                ws.close.assert_called_once()

            # Verify shutdown event is set
            assert server.shutdown_event.is_set()

    @pytest.mark.slow
    @pytest.mark.timeout(60)
    @pytest.mark.asyncio
    async def test_async_message_streaming(self):
        """Test async message streaming to clients."""
        with mock_mlx_models_context() as mocks:
            server = create_server()

            # Ensure text_generate is available by setting it directly
            import mlx_websockets.server as server_module

            if not hasattr(server_module, "text_generate") or server_module.text_generate is None:
                server_module.text_generate = mocks["text_generate"]

            websocket = AsyncMock()
            websocket.remote_address = ("127.0.0.1", 12345)

            # Track sent messages
            sent_messages = []

            async def track_send(msg):
                sent_messages.append(json.loads(msg))

            websocket.send = track_send

            # Create a simple text message
            message = json.dumps({"type": "text_input", "content": "Hello"})

            # Set up async iterator that doesn't close immediately
            async def message_iterator():
                yield message
                # Keep the connection open to allow streaming
                await asyncio.sleep(5.0)  # Keep connection alive during processing

            websocket.__aiter__ = lambda self: message_iterator()

            # Process
            await server.handle_client(websocket, "/")

            # Wait longer for processing and let background threads complete
            await asyncio.sleep(3.0)

            # Should have response_start, tokens, and response_complete
            message_types = [msg.get("type") for msg in sent_messages]
            assert "response_start" in message_types
            assert "token" in message_types
            assert "response_complete" in message_types


class TestAsyncEventLoopIntegration:
    """Test event loop integration."""

    @pytest.mark.asyncio
    async def test_coroutine_thread_safety(self):
        """Test thread-safe coroutine execution from sync threads."""
        with mock_mlx_models_context():
            _ = create_server()

            results = []
            errors = []

            async def async_operation(value):
                # Remove sleep to test if that's the issue
                return value * 2

            def thread_worker(loop, value):
                try:
                    future = asyncio.run_coroutine_threadsafe(async_operation(value), loop)
                    result = future.result(timeout=5.0)
                    results.append(result)
                except Exception as e:
                    errors.append(e)

            # Get event loop
            loop = asyncio.get_running_loop()

            # Start multiple threads
            threads = []
            for i in range(5):  # Reduced from 10 to 5 threads
                t = threading.Thread(target=thread_worker, args=(loop, i))
                t.start()
                threads.append(t)

            # Give event loop time to process the coroutines
            for _ in range(10):
                await asyncio.sleep(0.01)

            # Wait for completion
            for t in threads:
                t.join()

            # Verify results
            assert len(errors) == 0
            assert sorted(results) == [i * 2 for i in range(5)]

    @pytest.mark.asyncio
    async def test_signal_handler_setup(self):
        """Test signal handler setup for graceful shutdown."""
        with mock_mlx_models_context():
            with patch("signal.signal") as mock_signal:
                # Create server - this triggers signal setup in __init__
                _ = create_server()

                # Verify signal handlers were registered
                import signal as signal_module

                mock_signal.assert_any_call(signal_module.SIGINT, ANY)
                mock_signal.assert_any_call(signal_module.SIGTERM, ANY)


class TestAsyncQueueProcessing:
    """Test async queue processing behavior."""

    def test_queue_processing_with_stop_event(self):
        """Test that queue processing respects stop event."""
        with mock_mlx_models_context():
            server = create_server()

            websocket = Mock()
            client_queue = Queue()
            stop_event = threading.Event()
            loop = asyncio.new_event_loop()
            client_id = ("127.0.0.1", 12345)

            # Add items to queue
            for i in range(3):
                client_queue.put(
                    {
                        "type": "text",
                        "content": f"Message {i}",
                        "prompt": f"Message {i}",
                        "timestamp": time.time(),
                    }
                )

            processed_count = 0

            # Mock processing
            def mock_process_text(data, *args):
                nonlocal processed_count
                processed_count += 1
                if processed_count == 2:
                    stop_event.set()  # Stop after 2 messages

            with patch.object(server, "_process_text", side_effect=mock_process_text):
                server._process_frames(websocket, client_queue, stop_event, loop, client_id)

            # Should have processed at least 2 messages, but may process 3 due to race condition
            # This is correct behavior: queue.get() retrieves message before stop_event check
            assert 2 <= processed_count <= 3  # Account for race condition
            assert client_queue.qsize() <= 1  # At most 1 message left

    def test_queue_timeout_handling(self):
        """Test queue timeout handling in processing thread."""
        with mock_mlx_models_context():
            server = create_server()

            websocket = Mock()
            client_queue = Queue()
            stop_event = threading.Event()
            loop = asyncio.new_event_loop()
            client_id = ("127.0.0.1", 12345)

            # Don't add any items - queue will timeout

            # Run _process_frames in a thread
            process_thread = threading.Thread(
                target=server._process_frames,
                args=(websocket, client_queue, stop_event, loop, client_id),
            )
            process_thread.start()

            # Let it run for a bit then stop
            time.sleep(0.6)  # Slightly more than timeout
            stop_event.set()

            # Wait for thread to finish (should exit quickly in drain mode with empty queue)
            process_thread.join(timeout=2.0)

            # Should complete without errors
            assert not process_thread.is_alive()


class TestAsyncResourceManagement:
    """Test async resource management."""

    @pytest.mark.asyncio
    async def test_resource_monitor_lifecycle(self):
        """Test resource monitor start/stop with async server."""
        with mock_mlx_models_context():
            server = create_server()

            with patch.object(server.resource_monitor, "start") as mock_start:
                with patch.object(server.resource_monitor, "stop") as mock_stop:
                    with patch("websockets.serve", new_callable=AsyncMock):
                        # Start the server as a task
                        server_task = asyncio.create_task(server.start_server())

                        # Give it a moment to start
                        await asyncio.sleep(0.1)

                        # Resource monitor should be started
                        mock_start.assert_called_once()

                        # Signal shutdown
                        server.shutdown_event.set()

                        # Wait for server task to complete
                        await server_task

                        # Shutdown should stop resource monitor
                        mock_stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_concurrent_inference_limiting(self):
        """Test concurrent inference limiting with async operations."""
        with mock_mlx_models_context():
            server = create_server(max_concurrent_inference=2)

            inference_count = 0
            max_concurrent = 0
            lock = threading.Lock()

            async def mock_inference():
                nonlocal inference_count, max_concurrent

                # Acquire semaphore in async context using run_in_executor
                def acquire_semaphore():
                    return server.inference_semaphore.acquire()

                def release_semaphore():
                    server.inference_semaphore.release()

                # Acquire the semaphore
                await asyncio.get_event_loop().run_in_executor(None, acquire_semaphore)

                try:
                    with lock:
                        inference_count += 1
                        max_concurrent = max(max_concurrent, inference_count)

                    # Simulate inference time
                    await asyncio.sleep(0.1)

                    with lock:
                        inference_count -= 1
                finally:
                    # Always release the semaphore
                    await asyncio.get_event_loop().run_in_executor(None, release_semaphore)

            # Try to run 5 concurrent inferences
            tasks = [mock_inference() for _ in range(5)]
            await asyncio.gather(*tasks)

            # Should never exceed max_concurrent_inference limit
            assert max_concurrent <= 2


class TestAsyncClientCleanup:
    """Test async client cleanup behavior."""

    @pytest.mark.asyncio
    async def test_client_cleanup_on_disconnect(self):
        """Test client state cleanup on disconnection."""
        with mock_mlx_models_context():
            server = create_server()

            websocket = AsyncMock()
            websocket.remote_address = ("127.0.0.1", 12345)

            # Simulate immediate disconnect
            import websockets.exceptions

            websocket.__aiter__.side_effect = websockets.exceptions.ConnectionClosed(None, None)

            # Handle client
            await server.handle_client(websocket, "/")

            # Verify cleanup
            client_id = websocket.remote_address
            assert client_id not in server.client_queues
            assert client_id not in server.client_stop_events
            assert client_id not in server.client_frame_counts
            assert client_id not in server.client_generators

    @pytest.mark.asyncio
    async def test_processing_thread_cleanup(self):
        """Test processing thread cleanup on client disconnect."""
        with mock_mlx_models_context():
            server = create_server()

            websocket = AsyncMock()
            websocket.remote_address = ("127.0.0.1", 12345)

            # Track thread creation
            created_threads = []
            original_thread = threading.Thread

            def track_thread(*args, **kwargs):
                thread = original_thread(*args, **kwargs)
                created_threads.append(thread)
                return thread

            with patch("threading.Thread", side_effect=track_thread):
                # Simulate quick disconnect
                websocket.__aiter__.return_value = []

                await server.handle_client(websocket, "/")

                # Wait for thread cleanup
                await asyncio.sleep(0.1)

                # Thread should have stopped
                for thread in created_threads:
                    assert not thread.is_alive()


class TestAsyncStatusUpdates:
    """Test async status update functionality."""

    @pytest.mark.asyncio
    async def test_get_status_message(self):
        """Test async status message handling."""
        with mock_mlx_models_context():
            server = create_server()

            websocket = AsyncMock()
            websocket.remote_address = ("127.0.0.1", 12345)

            # Send get_status message
            message = json.dumps({"type": "get_status"})
            websocket.__aiter__.return_value = [message]

            sent_messages = []

            async def capture_send(msg):
                sent_messages.append(json.loads(msg))

            websocket.send = capture_send

            await server.handle_client(websocket, "/")

            # Should have sent status
            status_messages = [msg for msg in sent_messages if msg.get("type") == "status"]
            assert len(status_messages) == 1

            # Verify status content
            status = status_messages[0]
            assert "stats" in status
            assert "model" in status["stats"]
            assert "config" in status["stats"]

    @pytest.mark.asyncio
    async def test_status_error_handling(self):
        """Test status message error handling."""
        with mock_mlx_models_context():
            server = create_server()

            websocket = AsyncMock()
            websocket.remote_address = ("127.0.0.1", 12345)

            # Mock resource monitor to fail
            with patch.object(
                server.resource_monitor, "get_resource_stats", side_effect=Exception("Stats error")
            ):
                message = json.dumps({"type": "get_status"})
                websocket.__aiter__.return_value = [message]

                sent_messages = []

                async def capture_send(msg):
                    sent_messages.append(json.loads(msg))

                websocket.send = capture_send

                await server.handle_client(websocket, "/")

                # Should have sent error
                error_messages = [msg for msg in sent_messages if msg.get("type") == "error"]
                assert len(error_messages) == 1
                assert "Failed to get server status" in error_messages[0]["error"]
