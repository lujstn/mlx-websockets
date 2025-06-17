# Code Review Checklist

This checklist helps prevent common bugs and maintain code quality in the MLX WebSockets project.

## 1. Global Variable Checks

### ❌ WRONG - Will pass even if variable is None
```python
if 'variable' not in globals():
    raise Error("Variable not loaded")
```

### ✅ CORRECT - Checks if actually usable
```python
if variable is None or not callable(variable):
    raise Error("Variable not loaded or not callable")
```

**Why this matters:** `'variable' in globals()` returns `True` even when `variable = None`. This can lead to attempting to call `None` as a function.

## 2. Lazy Loading Pattern

### ❌ WRONG - Eager loading in constructor
```python
class Server:
    def __init__(self):
        self._load_heavy_resources()  # Blocks initialization
        self.ready = True
```

### ✅ CORRECT - Lazy loading on first use
```python
class Server:
    def __init__(self, load_on_init=True):
        self._loaded = False
        if load_on_init:
            self._ensure_loaded()

    def _ensure_loaded(self):
        if not self._loaded:
            self._load_heavy_resources()
            self._loaded = True

    def process_request(self):
        self._ensure_loaded()  # Load only when needed
        # ... process
```

**Why this matters:** Lazy loading improves startup time and allows better control in tests.

## 3. Test Mock Timing

### ❌ WRONG - Import before mock setup
```python
from mlx_websockets.server import MLXStreamingServer  # Imports with load=None

def test_something():
    with mock_mlx_models_context():  # Too late!
        server = MLXStreamingServer()
```

### ✅ CORRECT - Mock before import
```python
def test_something():
    with mock_mlx_models_context():
        from mlx_websockets.server import MLXStreamingServer
        server = MLXStreamingServer(load_model_on_init=False)
```

**Why this matters:** Python caches imported modules. Mocks must be in place before first import.

## 4. Defensive Checks

### ❌ WRONG - Assuming function exists
```python
result = some_function(data)  # May be None!
```

### ✅ CORRECT - Check before use
```python
if some_function is None:
    raise RuntimeError("Function not available")
result = some_function(data)
```

## 5. Error Messages

### ❌ WRONG - Generic error
```python
raise Error("Failed to load")
```

### ✅ CORRECT - Specific, actionable error
```python
raise ModelLoadError(
    "MLX dependencies not loaded. "
    "Call _import_dependencies() first or ensure proper test mocking."
)
```

## 6. State Management

### ❌ WRONG - No tracking of initialization state
```python
class Server:
    def __init__(self):
        self.load_model()

    def load_model(self):
        # May be called multiple times!
        self.model = expensive_load()
```

### ✅ CORRECT - Track initialization state
```python
class Server:
    def __init__(self):
        self._model_loaded = False
        self._ensure_model_loaded()

    def _ensure_model_loaded(self):
        if not self._model_loaded:
            self.model = expensive_load()
            self._model_loaded = True
```

## Review Process

When reviewing PRs, check for:

1. **Global checks using `in globals()`** - Flag for review
2. **Heavy operations in `__init__`** - Suggest lazy loading
3. **Missing defensive checks** - Ensure None checks before calling
4. **Test complexity** - If tests need complex setup, the code might need refactoring
5. **Import order dependencies** - Flag if tests are sensitive to import order

## Red Flags

- `if 'x' not in globals()` - Always wrong for our codebase
- Direct model/resource loading in constructors
- Tests that modify `sys.modules` without clear documentation
- Missing `is None` checks before calling functions
- Overly complex test fixtures (might indicate design issues)

## Best Practices

1. **Make lazy loading explicit** - Use flags like `load_on_init`
2. **Document mock requirements** - If tests need special setup, document why
3. **Fail fast with clear errors** - Better to catch issues early
4. **Test the actual code path** - Don't over-mock; test real behavior when possible
5. **Keep it simple** - Complex test setup often indicates complex code

Remember: If tests are hard to write, the code is probably hard to use. Fix the code, not the tests.
