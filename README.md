# mlx-websockets

A high-performance WebSocket server for streaming multimodal data (text, images, video) to MLX models running locally on Apple Silicon Macs, enabling real-time AI responses with minimal latency.

## Features

- 🚀 **Real-time streaming** - WebSocket-based communication for low-latency inference
- 🎯 **Designed for running local multimodal models** - Handle text, images, and video frames seamlessly (we default to Gemma 3 4-bit for this reason)
- 🔧 **MLX optimized** - Leverages Apple's MLX framework for efficient on-device inference
- 🎛️ **Flexible model support** - Use any compatible model from Hugging Face
- 🎬 **Stream processing** - Queue-based architecture with frame dropping for real-time performance
- 💬 **Token streaming** - Real-time token-by-token response streaming

## Requirements

- Mac with Apple Silicon
- macOS 13.0 (Ventura) or later
- Python 3.9+
- ~6GB available memory (for default Gemma 3 4-bit model)
- Xcode Command Line Tools (for MLX installation)

## Installation

1. **Clone the repository**:

```bash
git clone https://github.com/lujstn/mlx-websockets.git
cd mlx-websockets
```

2. **Install dependencies**:

```bash
pip install -r requirements.txt
```

3. **Verify MLX installation** _(optional)_:

```bash
python -c "import mlx; print(mlx.__version__)"
```

## Development

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

We use `black` for code formatting and `ruff` for linting:

```bash
# Format code
make format

# Check linting
make lint

# Run everything (format, lint, test)
make all
```

## Quick Start

```bash
python mlx_streaming_server.py
```

The server will start on `ws://localhost:8765` by default.

### Supported Input Types

- **Text** (`text_input`): Plain text messages with optional context
- **Images** (`image_input`): Base64 encoded images (JPEG, PNG) for single analysis
- **Video Frames** (`video_frame`/`screen_frame`): Continuous stream of frames for real-time analysis

**Note on Audio**: Audio inputs are not directly supported by current multimodal models. To process audio, you'll need to first transcribe it using a speech-to-text model (like Whisper) and then send the transcription as text input.

### Configuration

#### Command Line Arguments

Specify a different model or port using command-line arguments:

```bash
# Use a different model
python mlx_streaming_server.py --model "mlx-community/your-model-id"

# Change port
python mlx_streaming_server.py --port 8080

# Both
python mlx_streaming_server.py --model "mlx-community/your-model-id" --port 8080
```

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

**Note**: `presencePenalty` and `frequencyPenalty` are converted to MLX's `repetition_penalty` parameter for similar effect.

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
- **Token limits**: 200 tokens for text, 100 for images
- **Frame dropping**: Maintains real-time performance under load

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
   - Try a different port: `python mlx_streaming_server.py --port 8080`

### Memory Management

- The server automatically resizes images to max 768px to save memory
- Frame dropping prevents memory buildup during high-load scenarios
- Monitor GPU memory: The server prints memory usage on startup

## Model Compatibility

This server works with MLX-compatible models from Hugging Face:

- **Vision-Language Models**: Models that support both text and image inputs
- **Text-Only Models**: Automatically detected and handled via fallback
  - The server automatically falls back from vision API to text-only API when needed
  - This ensures compatibility with both multimodal and text-only MLX models
- **Recommended Models**:
  - `mlx-community/gemma-3-4b-it-4bit` (default)
  - `mlx-community/llama-3.2-11b-vision-instruct-4bit`

## Security Considerations

- By default, the server only accepts connections from localhost
- For network access, modify the server binding in `start_server()`
- Always use WSS (WebSocket Secure) for production deployments
- Implement authentication before exposing to networks

## Architecture

The server uses a multi-threaded, queue-based architecture:

1. **Main Thread**: Handles WebSocket connections and async I/O
2. **Processing Threads**: One per client for model inference
3. **Queue System**: Buffers requests with automatic frame dropping
4. **Thread Safety**: All shared resources protected by locks
   - `RLock` for client state and configuration (allows re-entrant locking)
   - `Lock` for model inference (ensures single inference at a time)
   - Thread-safe message passing between async and sync contexts

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Credits

🏃 Built by Lucas Johnston Kurilov ([lujstn.com](https://lujstn.com), [@lujstn](https://tiktok.com/@lujstn))

_This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details._

- Built on [Apple MLX](https://github.com/ml-explore/mlx) - Fast machine learning framework for Apple Silicon
- Default model: [Gemma 3 4-bit](https://huggingface.co/mlx-community/gemma-3-4b-it-4bit) by Google, quantized by the MLX Community
- WebSocket implementation using [websockets](https://github.com/python-websockets/websockets) library
