# Testing Strategy for MLX WebSocket Server

## Overview

The MLX WebSocket Server uses a hybrid async/threading architecture:
- Async WebSocket handling in the main thread
- Background thread per client for ML inference
- Thread-safe communication via queues and asyncio.run_coroutine_threadsafe

## Current Issues with Tests

1. **Over-mocking**: Tests mock threading.Thread entirely, preventing real behavior testing
2. **Async/Sync Confusion**: Mixing async mocks with sync thread execution
3. **No Real Integration**: Tests don't verify the actual async↔thread communication

## Recommended Testing Approach

### 1. Unit Tests (test_server.py, test_inference.py)

**Goal**: Test business logic in isolation

**Strategy**:
- Mock ML models (load, generate functions)
- Don't mock threading - let the real threads run
- Use synchronous test helpers for thread communication
- Focus on logic correctness, not performance

**Example Pattern**:
```python
@pytest.mark.asyncio
async def test_text_processing():
    with mock_mlx_models() as mocks:
        server = MLXStreamingServer()
        
        # Create a proper async mock websocket
        websocket = create_async_websocket_mock()
        
        # Run the full handle_client flow
        await server.handle_client(websocket, "/")
        
        # Verify the logic worked correctly
        assert_correct_responses(websocket.sent_messages)
```

### 2. Integration Tests (test_integration.py)

**Goal**: Test end-to-end flows with real async/thread interaction

**Strategy**:
- Mock only external dependencies (ML models)
- Use real WebSocket-like objects with proper async behavior
- Test actual message flows and error handling
- Verify thread cleanup and resource management

**Example Pattern**:
```python
@pytest.mark.asyncio
async def test_full_conversation():
    async with create_test_server() as server:
        async with create_test_client(server.url) as client:
            # Send real messages
            await client.send(json.dumps({...}))
            
            # Receive real responses
            response = await client.recv()
            
            # Verify end-to-end behavior
            assert json.loads(response)["type"] == "response_complete"
```

### 3. Benchmark Tests (test_benchmarks.py)

**Goal**: Measure performance of core operations

**Strategy**:
- Mock ML models but measure real processing time
- Focus on throughput of the processing logic
- Don't measure threading overhead in unit benchmarks
- Create separate integration benchmarks for full-stack performance

**Example Pattern**:
```python
def benchmark_token_processing():
    # Direct benchmark of the processing logic
    tokens = generate_test_tokens(1000)
    
    start = time.time()
    for token in tokens:
        process_token(token)
    end = time.time()
    
    throughput = len(tokens) / (end - start)
    assert throughput > EXPECTED_MIN_THROUGHPUT
```

### 4. Thread Safety Tests

**Goal**: Verify concurrent access is safe

**Strategy**:
- Use threading but with controlled execution
- Test race conditions and edge cases
- Verify proper cleanup on errors

**Example Pattern**:
```python
def test_concurrent_client_handling():
    server = MLXStreamingServer()
    
    # Create multiple threads accessing shared state
    threads = []
    for i in range(10):
        t = threading.Thread(target=lambda: server.handle_client(...))
        threads.append(t)
        t.start()
    
    # Wait and verify no corruption
    for t in threads:
        t.join()
    
    assert_server_state_consistent(server)
```

## Key Principles

1. **Mock at the Right Level**: Mock external dependencies (ML models), not internal implementation (threading)

2. **Respect Async/Sync Boundaries**: Don't mix async mocks with sync execution

3. **Test What Matters**: 
   - Unit tests: Logic correctness
   - Integration tests: End-to-end flows
   - Benchmarks: Performance characteristics

4. **Clean Up Properly**: Always ensure threads are stopped and resources freed

5. **Avoid Timing Dependencies**: Use events/queues instead of sleep() where possible

## Implementation Steps

1. Update test helpers to provide proper async/sync mocks
2. Refactor tests to follow the patterns above
3. Add thread lifecycle tracking for better debugging
4. Create separate performance harness for full-stack benchmarks
5. Document expected behavior for each test type

## Benefits

- Tests actually verify real behavior
- Easier to debug failures
- Better confidence in production behavior
- Clearer separation of concerns
- More maintainable test suite