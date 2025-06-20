# mlx-websockets

A high-performance WebSocket server for streaming multimodal data (text, images, video) to MLX models running locally on Apple Silicon Macs, enabling real-time AI responses with minimal latency.

## Installation

```bash
brew tap lujstn/tap
brew install mlx-websockets
```

## Features

- 🚀 **Real-time streaming** - WebSocket-based communication for low-latency inference
- 🎯 **Designed for running local multimodal models** - Handle text, images, and video frames seamlessly (we default to Gemma 3 4-bit for this reason)
- 🔧 **MLX optimized** - Leverages Apple's MLX framework for efficient on-device inference
- 🎛️ **Flexible model support** - Use any compatible model from Hugging Face
- 🎬 **Stream processing** - Queue-based architecture with frame dropping for real-time performance
- 💬 **Token streaming** - Real-time token-by-token response streaming
- 🎨 **Rich terminal UI** - Colorful output with progress bars (optional)
- 🛡️ **Graceful shutdown** - Clean termination with SIGINT/SIGTERM handling
- 🔍 **Debug mode** - Comprehensive logging for development
- 🌐 **Network discovery** - Automatic detection of all network interfaces

## Requirements

- Mac with Apple Silicon
- macOS 13.0 (Ventura) or later
- Python 3.9+
- ~6GB available memory (for default Gemma 3 4-bit model)
- Xcode Command Line Tools (for MLX installation)

## Installation

### From Source

1. **Clone the repository**:

```bash
git clone https://github.com/lujstn/mlx-websockets.git
cd mlx-websockets
```

2. **Install the package**:

```bash
pip install -e .
```

3. **Verify installation** _(optional)_:

```bash
mlx help
```

## Development

### Quick Start for Contributors

1. Clone and enter the repository:
   ```bash
   git clone https://github.com/lujstn/mlx-websockets.git
   cd mlx-websockets
   ```

2. Set up your development environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ./scripts/setup.sh
   ```

   The setup script will install all dependencies and configure git hooks that run comprehensive checks before commits and pushes to catch issues early.

### Testing

The project includes a comprehensive test suite using pytest:

```bash
# Run all tests
make test

# Run tests with coverage
make test-cov

# Run specific test file
pytest tests/test_server.py -v
```

### Linting and Formatting

We use `ruff` for both code formatting and linting:

```bash
# Format code
make format

# Check linting
make lint

# Run everything (format, lint, test)
make all
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed development guidelines.

## Quick Start

```bash
mlx serve
```

The server will start on `ws://localhost:8765` by default and display all available network addresses for connection. The startup includes:

- Colorful terminal UI with progress bars (if `rich` is installed)
- Model loading with memory usage display
- Network interface discovery showing all connection URLs
- Automatic suppression of harmless warnings from dependencies

### Running in Background

```bash
# Start server in background
mlx background serve

# Check status
mlx status

# Stop background server
mlx background stop
```

### Supported Input Types

- **Text** (`text_input`): Plain text messages with optional context
- **Images** (`image_input`): Base64 encoded images (JPEG, PNG) for single analysis
- **Video Frames** (`video_frame`/`screen_frame`): Continuous stream of frames for real-time analysis

**Note on Audio**: Audio inputs are not directly supported by current multimodal models. To process audio, you'll need to first transcribe it using a speech-to-text model (like Whisper) and then send the transcription as text input.

### Configuration

#### Command Line Arguments

Configure the server at startup using command-line arguments:

| Argument              | Type   | Default                            | Description                                               |
| --------------------- | ------ | ---------------------------------- | --------------------------------------------------------- |
| `--model`             | string | `mlx-community/gemma-3-4b-it-4bit` | Hugging Face model ID (any MLX-compatible model)          |
| `--port`              | int    | `8765`                             | WebSocket server port                                     |
| `--auto-port`         | flag   | `False`                            | Enable automatic port discovery if preferred port is busy |
| `--host`              | string | `0.0.0.0`                          | Host to bind to (use 'localhost' for local-only)          |
| `--trust-remote-code` | flag   | `False`                            | Enable trusting remote code for tokenizer                 |
| `--tokenizer-config`  | string | None                               | Tokenizer config.json file path                           |
| `--chat-template`     | string | None                               | Chat template or template name                            |
| `--max-tokens`        | int    | `512`                              | Maximum number of tokens to generate                      |
| `--temperature`       | float  | `0.7`                              | Sampling temperature                                      |
| `--seed`              | int    | None                               | Random seed for generation                                |
| `--debug`             | flag   | `False`                            | Enable debug logging                                      |
| `--show-warnings`     | flag   | `False`                            | Show all warnings (including harmless ones)               |

Example usage:

```bash
# Use a different model
mlx serve --model "mlx-community/Phi-3.5-mini-instruct-4bit"

# Try Qwen for better multilingual support
mlx serve --model "mlx-community/Qwen2.5-3B-Instruct-4bit"

# Use a larger Gemma model for better quality
mlx serve --model "mlx-community/gemma-2-9b-it-4bit"

# Change port with auto-discovery
mlx serve --port 8080 --auto-port

# Enable debug mode
mlx serve --debug

# Local-only access
mlx serve --host localhost

# Configure generation parameters
mlx serve --max-tokens 1000 --temperature 0.9 --seed 42

# Trust remote code for specialized models
mlx serve --model "custom-model" --trust-remote-code

# Multiple options
mlx serve --model "mlx-community/your-model-id" --port 8080 --debug --auto-port

# Run in background
mlx background serve
```

**Note**: The background server automatically finds an available port if the default (8765) is in use.

#### Runtime Configuration

You can update generation parameters at runtime via WebSocket. All fields are optional:

| Parameter            | Type    | Default  | Description                                                     |
| -------------------- | ------- | -------- | --------------------------------------------------------------- |
| `candidateCount`     | integer | 1        | Number of responses to generate (MLX typically supports only 1) |
| `maxOutputTokens`    | integer | 200      | Maximum tokens in response                                      |
| `temperature`        | number  | 0.7      | Controls randomness (0.0-2.0). Lower = more deterministic       |
| `topP`               | number  | 1.0      | Cumulative probability threshold for token sampling (0.0-1.0)   |
| `topK`               | integer | 50       | Maximum number of tokens to consider when sampling              |
| `presencePenalty`    | number  | 0.0      | Penalty for using tokens that appear in the response            |
| `frequencyPenalty`   | number  | 0.0      | Penalty scaled by token usage frequency                         |
| `responseModalities` | array   | ["TEXT"] | Output types (currently only TEXT is supported)                 |
| `max_tokens_image`   | integer | 100      | Special: max tokens for image inputs                            |

**Note**: `presencePenalty` and `frequencyPenalty` are converted to MLX's `repetition_penalty` parameter for similar effect. The conversion maps penalty range [0, 2] to repetition_penalty range [1.0, 1.5], using the stronger of the two penalties when both are specified. The repetition context size is fixed at 20 tokens.

Example configuration update:

```json
{
  "type": "config",
  "temperature": 0.9,
  "maxOutputTokens": 300
}
```

## API Reference

### Input Messages

#### Text Input

```json
{
  "type": "text_input",
  "content": "Your question or prompt",
  "context": "Optional context" // optional
}
```

#### Image Input

```json
{
  "type": "image_input",
  "image": "data:image/png;base64,<base64_data>", // or just base64 string
  "prompt": "What do you see?",
  "source": "screenshot" // optional, for tracking
}
```

#### Video/Screen Frame

```json
{
  "type": "video_frame", // or "screen_frame"
  "frame": "data:image/jpeg;base64,<base64_data>",
  "prompt": "Describe the scene",
  "source": "webcam" // optional
}
```

#### Configuration Update

```json
{
  "type": "config",
  // All fields are optional - provide only what you want to change
  "candidateCount": 1,
  "maxOutputTokens": 200,
  "temperature": 0.7,
  "topP": 0.9,
  "topK": 50,
  "presencePenalty": 0.0,
  "frequencyPenalty": 0.0,
  "responseModalities": ["TEXT"],
  "max_tokens_image": 100
}
```

### Response Messages

#### Response Start

```json
{
  "type": "response_start",
  "timestamp": 1234567890.123,
  "input_type": "text|image"
}
```

#### Token Stream

```json
{
  "type": "token",
  "content": "generated token",
  "timestamp": 1234567890.123
}
```

#### Response Complete

```json
{
  "type": "response_complete",
  "full_text": "Complete generated response",
  "timestamp": 1234567890.123,
  "input_type": "text|image",
  "inference_time": 0.456 // Time taken for inference in seconds
}
```

#### Frame Dropped (streaming only)

```json
{
  "type": "frame_dropped",
  "reason": "processing_queue_full"
}
```

#### Error

```json
{
  "type": "error",
  "error": "Error message",
  "timestamp": 1234567890.123
}
```

#### Configuration Updated

```json
{
  "type": "config_updated",
  "updated_fields": {
    // Only the fields that were changed
    "temperature": 0.9,
    "maxOutputTokens": 300
  },
  "current_config": {
    // Full current configuration
    "candidateCount": 1,
    "maxOutputTokens": 300,
    "temperature": 0.9,
    "topP": 1.0,
    "topK": 50,
    "presencePenalty": 0.0,
    "frequencyPenalty": 0.0,
    "responseModalities": ["TEXT"]
  }
}
```

## Performance

Optimized for Apple Silicon with Gemma 3's 4-bit model:

- **Memory**: ~4.5-6GB total (2.6GB model weights + 2-4GB runtime)
- **Processing**: Queue-based with max 10 frames buffered
- **Image sizing**: Auto-resizes to 768px max dimension
- **Token limits**: 200 tokens for text, 100 for images (configurable)
- **Frame dropping**: Maintains real-time performance under load
- **Timeouts**: 60s max inference, 5s WebSocket send, 2s thread shutdown
- **Performance tracking**: Per-client frame counting and inference time measurement
- **Memory display**: Shows active GPU memory usage on startup

## Troubleshooting

### Common Issues

1. **Model loading fails:**

   - Ensure you have enough free memory (~6GB)
   - Check internet connection for model download
   - Try clearing MLX cache: `rm -rf ~/.cache/huggingface/hub/`

2. **Import errors:**

   - Verify you're using Python 3.9+
   - Ensure virtual environment is activated
   - Reinstall MLX: `pip install --upgrade mlx mlx-lm mlx-vlm`

3. **WebSocket connection issues:**
   - Check if port 8765 is already in use
   - Try a different port: `mlx serve --port 8080`
   - The background server automatically finds an available port if the default is in use

### Memory Management

- The server automatically resizes images to max 768px to save memory
- Frame dropping prevents memory buildup during high-load scenarios
- Monitor GPU memory: The server prints memory usage on startup

## Model Compatibility

This server works with MLX-compatible models from Hugging Face. Look for models in the `mlx-community` namespace that typically end in `-4bit`.

- **Vision-Language Models**: Models that support both text and image inputs
- **Text-Only Models**: Automatically detected and handled via fallback
  - The server automatically falls back from vision API to text-only API when needed
  - This ensures compatibility with both multimodal and text-only MLX models
- **Popular Models**:
  - `mlx-community/gemma-3-4b-it-4bit` (default - balanced performance)
  - `mlx-community/Phi-3.5-mini-instruct-4bit` (faster, smaller)
  - `mlx-community/Qwen2.5-3B-Instruct-4bit` (great multilingual support)
  - `mlx-community/llama-3.2-11b-vision-instruct-4bit` (larger, higher quality)

## Security Considerations

- By default, the server binds to `0.0.0.0` (all interfaces)
- For local-only access, use `--host localhost`
- The server displays all available network addresses on startup
- Always use WSS (WebSocket Secure) for production deployments
- Implement authentication before exposing to networks

## Architecture

The server uses a multi-threaded, queue-based architecture:

1. **Main Thread**: Handles WebSocket connections and async I/O
2. **Processing Threads**: One per client for model inference
3. **Queue System**: Buffers requests with automatic frame dropping (max 10 items)
4. **Thread Safety**: All shared resources protected by locks
   - `RLock` for client state and configuration (allows re-entrant locking)
   - `Lock` for model inference (ensures single inference at a time)
   - Thread-safe message passing between async and sync contexts
5. **Client Management**: Per-client state tracking with:
   - Dedicated message queues
   - Stop events for graceful shutdown
   - Frame counters for performance monitoring
   - Active generator tracking for cleanup
6. **Graceful Shutdown**:
   - Signal handlers for SIGINT/SIGTERM
   - Active connection tracking and closure
   - Processing thread termination with timeout
   - MLX cache clearing and garbage collection

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Credits

🏃 Built by Lucas Johnston Kurilov ([lujstn.com](https://lujstn.com), [@lujstn](https://tiktok.com/@lujstn))

_This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details._

- Built on [Apple MLX](https://github.com/ml-explore/mlx) - Fast machine learning framework for Apple Silicon
- Default model: [Gemma 3 4-bit](https://huggingface.co/mlx-community/gemma-3-4b-it-4bit) by Google, quantized by the MLX Community
- WebSocket implementation using [websockets](https://github.com/python-websockets/websockets) library
