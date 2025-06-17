#!/usr/bin/env python3
"""
MLX WebSocket Streaming Server ⚡️
Real-time multimodal AI inference server for Apple Silicon Macs.
Supports streaming video frames, images, and text chat with local MLX models.
"""

import asyncio
import base64
import io
import json
import logging
import os
import signal
import socket
import sys
import threading
import time
import warnings
from collections import OrderedDict
from functools import lru_cache
from queue import Empty, Queue
from threading import Lock, RLock, Semaphore
from typing import Optional

import websockets
import websockets.exceptions

from .exceptions import (
    ClientConnectionError,
    ConfigurationError,
    ImageProcessingError,
    MessageProcessingError,
    ModelLoadError,
    NetworkError,
    ResourceError,
    TextGenerationError,
)
from .logging_utils import (
    ANSI_COLORS,
    get_logger,
    log_activity,
    log_error,
    log_info,
    log_success,
    log_warning,
    setup_logging,
)
from .process_registry import ProcessInfo, get_registry
from .resource_monitor import ResourceMonitor, check_memory_available

# Suppress warnings from dependencies
warnings.filterwarnings("ignore", message="Disabling PyTorch because PyTorch >= 2.1 is required")
warnings.filterwarnings(
    "ignore", message="None of PyTorch, TensorFlow >= 2.0, or Flax have been found"
)
warnings.filterwarnings(
    "ignore",
    message="The torchvision.datapoints and torchvision.transforms.v2 namespaces are still Beta",
)
warnings.filterwarnings("ignore", message="`resume_download` is deprecated")
warnings.filterwarnings("ignore", message="resource_tracker: There appear to be")
warnings.filterwarnings("ignore", category=UserWarning, module="multiprocessing.resource_tracker")

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning:multiprocessing.resource_tracker"

logger = logging.getLogger(__name__)

console = None  # type: Optional[Console]
try:
    from rich.console import Console
    from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
    from rich.text import Text
    from rich.theme import Theme

    RICH_AVAILABLE = True

    # Custom color theme
    custom_theme = Theme(
        {
            "white": "#fffbeb",  # Primary text and important information
            "cream": "#faf6e6",  # Softer white for loading text
            "dim_cream": "#E3DDDD",
            "pink": "#ff79c6",  # Accents, emphasis, and special highlights
            "purple": "#a07ae1",  # Secondary information and decorative elements
            "bold_purple": "bold #a07ae1",
            "yellow": "#ffd230",  # Warnings and transitional states
            "bold_yellow": "bold #ffd230",
            "amber": "#ffd230",  # Secondary warnings
            "cyan": "#00d3f2",  # Interactive elements, URLs, and key values
            "green": "#50fa7b",  # Keep default green for success
            "red": "#ff5555",  # Keep default red for errors
            "dim": "#a07ae1",  # Use purple for dim/secondary text
            "bold_white": "bold #fffbeb",  # Bold white text
        }
    )

    console = Console(theme=custom_theme)
except ImportError:
    RICH_AVAILABLE = False
    console = None


def create_elapsed_time_column():
    """Create a custom elapsed time column with colourful styling"""
    from rich.progress import ProgressColumn

    class ColourElapsedColumn(ProgressColumn):
        """Custom elapsed time column with colourful styling"""

        def render(self, task):
            colour = "cream"
            elapsed = task.finished_time if task.finished else task.elapsed
            if elapsed is None:
                return Text("-:--:--", style=colour)
            else:
                # Convert to timedelta and format
                import datetime

                td = datetime.timedelta(seconds=max(0, int(elapsed)))
                return Text(str(td), style=colour)

    return ColourElapsedColumn()


_dependencies_loaded = False
mx = None
generate = None
load = None
Image = None
text_generate = None


def _import_dependencies(debug=False):
    """Import MLX and other heavy dependencies with loading indicator"""
    global mx, websockets, generate, load, Image, text_generate, _dependencies_loaded

    if _dependencies_loaded:
        return

    if not debug and RICH_AVAILABLE:
        with Progress(
            SpinnerColumn(style="purple"),
            TextColumn("[cream][progress.description]{task.description}[/cream]"),
            BarColumn(pulse_style="purple"),
            create_elapsed_time_column(),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task("Loading PyTorch and dependencies...", total=None)

            import mlx.core as mx

            progress.update(task, description="Loading WebSockets...")
            progress.update(task, description="Loading MLX models...")

            from mlx_vlm import generate as gen  # type: ignore
            from mlx_vlm import load as ld

            generate = gen
            load = ld
            progress.update(task, description="Loading PIL...")

            from PIL import Image as img

            Image = img
            progress.update(task, description="Loading optional text generation...")

            try:
                from mlx_lm import generate as text_gen

                text_generate = text_gen
            except ImportError:
                text_generate = None

            progress.update(task, description="Dependencies loaded!", completed=True)
    else:
        import mlx.core as mx
        from mlx_vlm import generate as gen
        from mlx_vlm import load as ld

        generate = gen
        load = ld
        from PIL import Image as img

        Image = img

        try:
            from mlx_lm import generate as text_gen

            text_generate = text_gen
        except ImportError:
            text_generate = None

    _dependencies_loaded = True


class WebSocketHandshakeFilter(logging.Filter):
    """Filter to suppress verbose WebSocket handshake errors."""

    def __init__(self, server_instance):
        super().__init__()
        self.server = server_instance
        self._shown_hint = False

    def filter(self, record):
        # Check if this is a websockets handshake error
        if record.name.startswith("websockets") and record.levelno >= logging.ERROR:
            msg_str = str(record.getMessage()).lower()
            exc_text = str(getattr(record, "exc_text", "")).lower()

            # Check for handshake-related errors
            if any(
                pattern in msg_str or pattern in exc_text
                for pattern in [
                    "opening handshake failed",
                    "connection closed while reading",
                    "stream ends after 0 bytes",
                    "invalidmessage",
                    "eoferror",
                    "did not receive a valid http request",
                ]
            ):
                # Show a simple message instead of the full traceback
                if not self._shown_hint or self.server.debug:
                    # Use amber color for info messages
                    if RICH_AVAILABLE:
                        console.print(
                            "[amber]ℹ️  Non-WebSocket connection detected (likely a port scanner or health check)[/amber]"
                        )
                    else:
                        sys.stdout.write(
                            f"{ANSI_COLORS['amber']}ℹ️  Non-WebSocket connection detected (likely a port scanner or health check){ANSI_COLORS['reset']}\n"
                        )
                        sys.stdout.flush()
                    if not self._shown_hint:
                        self._shown_hint = True
                return False  # Suppress the original error
        return True


class CustomWebSocketLogger(logging.LoggerAdapter):
    """Custom logger adapter for websockets that provides cleaner error messages."""

    def __init__(self, logger, server):
        super().__init__(logger, {})
        self.server = server

    def error(self, msg, *args, **kwargs):
        # Intercept error messages and check if they're handshake-related
        if "opening handshake failed" in str(msg).lower():
            # Use info level for these expected errors
            self.info("Connection attempt rejected (not a valid WebSocket request)")
            return
        super().error(msg, *args, **kwargs)


class MLXStreamingServer:
    def __init__(
        self,
        model_name="mlx-community/gemma-3-4b-it-4bit",
        port=8765,
        host="0.0.0.0",
        debug=False,
        max_concurrent_inference=2,
        enable_response_cache=True,
        cache_size=100,
        load_model_on_init=True,
        auto_port=True,
    ):
        """
        Initialize the MLX WebSocket Streaming Server

        Args:
            model_name: MLX model from HuggingFace (Gemma 3 4bit model recommended)
            port: WebSocket server port (default 8765, 0 for OS assignment)
            host: Host to bind to (default 0.0.0.0 for all interfaces)
            debug: Enable debug logging (default False)
            max_concurrent_inference: Maximum concurrent model inferences (default 2)
            enable_response_cache: Enable response caching for repeated queries (default True)
            cache_size: Maximum number of cached responses (default 100)
            load_model_on_init: Whether to load model immediately (default True for backwards compatibility)
            auto_port: Enable automatic port discovery if preferred port is busy (default True)
        """
        self.model_name = model_name
        self.requested_port = port
        self.port = port  # For backward compatibility
        self.auto_port = auto_port
        self.actual_port = None  # Set during start_server
        self.host = host
        self.debug = debug
        self.model = None
        self.processor = None
        self.max_concurrent_inference = max_concurrent_inference
        self.enable_response_cache = enable_response_cache

        # Thread safety
        self.clients_lock = RLock()  # Protects client dictionaries
        self.config_lock = RLock()  # Protects configuration
        self.inference_semaphore = Semaphore(
            max_concurrent_inference
        )  # Allows concurrent inference
        self.cache_lock = Lock()  # Protects response cache

        # Client state management (protected by clients_lock)
        self.client_queues = {}
        self.client_stop_events = {}  # Stop events for thread termination
        self.client_frame_counts = {}  # Per-client frame counting
        self.client_generators = {}  # Track active generators for cleanup

        # Response cache (protected by cache_lock)
        if enable_response_cache:
            self.response_cache = OrderedDict()  # LRU cache implementation
            self.cache_size = cache_size
            self.cache_hits = 0
            self.cache_misses = 0
        else:
            self.response_cache = None

        # Batch queue for future batch processing support
        self.batch_queue = Queue(maxsize=50)
        self.batch_processing_thread = None

        # Performance metrics
        self.total_requests = 0
        self.total_inference_time = 0.0
        self.concurrent_inferences = 0
        self.max_concurrent_reached = 0

        # Message processing callbacks for testing/monitoring
        self._message_processed_callback = None

        # Default configuration in MLX-native format (protected by config_lock)
        self.config = {
            # MLX-native parameters
            "maxTokens": 200,  # Maximum tokens to generate
            "temperature": 0.7,  # Randomness (0.0-1.0)
            "topP": 1.0,  # Nucleus sampling threshold
            "topK": 50,  # Top-k sampling
            "repetitionPenalty": 1.0,  # Penalty for token repetition (1.0 = no penalty)
            "repetitionContextSize": 20,  # Tokens to consider for repetition
            "seed": None,  # Random seed for reproducibility
            # Metadata
            "candidateCount": 1,  # MLX only supports single response
            "responseModalities": ["TEXT"],  # Currently only TEXT is supported
        }

        # Separate max tokens for different input types (convenience)
        # Protected by config_lock since it's configuration
        self.max_tokens_image = 100  # Images typically need fewer tokens

        # Shutdown handling
        self.shutdown_event = asyncio.Event()
        self.shutting_down = False  # Initialize shutdown flag
        self.active_connections = set()  # Track active WebSocket connections
        self.server = None  # Will hold the WebSocket server
        self._cleanup_registered = False
        self._register_cleanup()

        # Resource monitoring
        self.resource_monitor = ResourceMonitor()

        # Model loading state
        self._model_loaded = False

        # Check memory before loading model
        mem_ok, mem_msg = check_memory_available(required_gb=4.0)
        if not mem_ok:
            log_warning(mem_msg)

        # Load model if requested
        if load_model_on_init:
            self._ensure_model_loaded()

    def _ensure_model_loaded(self):
        """Ensure model is loaded before use."""
        if not self._model_loaded:
            if not self.debug:
                log_info(f"Loading model: {self.model_name}")
            else:
                logger.info(f"Loading model: {self.model_name}")
            self._load_model()
            self._model_loaded = True

    def _load_model(self):
        """Load the MLX model and processor"""
        # Check if dependencies are available
        if load is None or not callable(load):
            error_msg = "MLX dependencies not loaded. Call _import_dependencies() first or ensure proper test mocking."
            if not self.debug:
                log_error(f"Dependency error: {error_msg}")
            else:
                logger.error(f"Dependency error: {error_msg}")
            raise ModelLoadError(error_msg)

        try:
            # Capture warnings during model loading
            with warnings.catch_warnings(record=True) as caught_warnings:
                warnings.simplefilter("always")

                if not self.debug and RICH_AVAILABLE:
                    with Progress(
                        SpinnerColumn(style="pink"),
                        TextColumn("[cream][progress.description]{task.description}[/cream]"),
                        BarColumn(pulse_style="pink"),
                        create_elapsed_time_column(),
                        console=console,
                        transient=True,
                    ) as progress:
                        task = progress.add_task(
                            "[cream]Downloading model files...[/cream]", total=None
                        )
                        self.model, self.processor = load(self.model_name)
                        progress.update(task, description="Model loaded!", completed=True)
                else:
                    self.model, self.processor = load(self.model_name)

                # Show non-critical warnings in grey if any
                if caught_warnings and self.debug:
                    for w in caught_warnings:
                        if "Xet Storage" in str(w.message):
                            continue  # Skip Xet storage warnings
                        logger.debug(f"Warning: {w.message}")

            if not self.debug:
                log_success("Model loaded successfully!")
                if mx is not None:
                    mlx_memory = mx.get_active_memory() / 1024**3
                    log_info(f"MLX memory usage: {mlx_memory:.2f} GB")
            else:
                logger.info("Model loaded successfully!")
                logger.info(f"Memory usage: {mx.get_active_memory() / 1024**3:.2f} GB")
        except FileNotFoundError as e:
            error_msg = (
                f"Model '{self.model_name}' not found. Please check the model name and try again."
            )
            if not self.debug:
                log_error(f"Model not found: {error_msg}")
            else:
                logger.error(f"Model not found: {e}")
            raise ModelLoadError(error_msg) from e
        except ImportError as e:
            error_msg = "Missing required dependencies. Please install mlx-vlm and mlx-lm."
            if not self.debug:
                log_error(f"Import error: {error_msg}")
            else:
                logger.error(f"Import error: {e}")
            raise ModelLoadError(error_msg) from e
        except MemoryError as e:
            error_msg = "Insufficient memory to load model. Try a smaller model or free up memory."
            if not self.debug:
                log_error(f"Memory error: {error_msg}")
            else:
                logger.error(f"Memory error: {e}")
            raise ModelLoadError(error_msg) from e
        except Exception as e:
            error_msg = f"Failed to load model: {str(e)}"
            if not self.debug:
                log_error(f"Error loading model: {error_msg}")
            else:
                logger.error(f"Error loading model: {e}", exc_info=True)
            raise ModelLoadError(error_msg) from e

    def _get_cache_key(self, prompt: str, config: dict) -> str:
        """Generate a cache key for the given prompt and config"""
        # Create a hashable key from prompt and relevant config
        key_parts = [
            prompt,
            str(config.get("temperature", 0.7)),
            str(config.get("maxTokens", 200)),
            str(config.get("topP", 1.0)),
            str(config.get("topK", 50)),
            str(config.get("repetitionPenalty", 1.0)),
            str(config.get("seed", "None")),
        ]
        return "|".join(key_parts)

    def _get_cached_response(self, cache_key: str) -> Optional[str]:
        """Get cached response if available"""
        if self.response_cache is None:
            return None

        with self.cache_lock:
            if cache_key in self.response_cache:
                # Move to end (LRU)
                self.response_cache.move_to_end(cache_key)
                self.cache_hits += 1
                return self.response_cache[cache_key]
            else:
                self.cache_misses += 1
                return None

    def _cache_response(self, cache_key: str, response: str):
        """Cache a response"""
        if self.response_cache is None:
            return

        with self.cache_lock:
            # Add to cache
            self.response_cache[cache_key] = response

            # Evict oldest if cache is full
            if len(self.response_cache) > self.cache_size:
                self.response_cache.popitem(last=False)

    def _update_performance_metrics(self, inference_time: float):
        """Update performance metrics"""
        with self.clients_lock:
            self.total_requests += 1
            self.total_inference_time += inference_time

            if self.concurrent_inferences > self.max_concurrent_reached:
                self.max_concurrent_reached = self.concurrent_inferences

            # Log performance stats every 100 requests
            if self.total_requests % 100 == 0:
                avg_time = self.total_inference_time / self.total_requests
                cache_hit_rate = (
                    (self.cache_hits / (self.cache_hits + self.cache_misses) * 100)
                    if self.response_cache and (self.cache_hits + self.cache_misses) > 0
                    else 0
                )
                logger.info(
                    f"Performance stats - Requests: {self.total_requests}, "
                    f"Avg inference time: {avg_time:.2f}s, "
                    f"Max concurrent: {self.max_concurrent_reached}, "
                    f"Cache hit rate: {cache_hit_rate:.1f}%"
                )

    async def handle_client(self, websocket, path):
        """Handle WebSocket client connections"""
        client_id = websocket.remote_address
        if self.debug:
            logger.info(f"Client connected: {client_id}")
        else:
            log_info(f"Client connected: {client_id[0]}:{client_id[1]}")

        self.active_connections.add(websocket)

        client_queue = Queue(maxsize=10)
        stop_event = threading.Event()

        with self.clients_lock:
            self.client_queues[client_id] = client_queue
            self.client_stop_events[client_id] = stop_event
            self.client_frame_counts[client_id] = 0
            self.client_generators[client_id] = []

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.get_event_loop()

        processing_thread = threading.Thread(
            target=self._process_frames,
            args=(websocket, client_queue, stop_event, loop, client_id),
            daemon=True,
        )
        processing_thread.start()

        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                except json.JSONDecodeError as e:
                    error_msg = "Invalid message format. Please send valid JSON."
                    if self.debug:
                        logger.error(f"JSON decode error from client {client_id}: {e}")
                    else:
                        log_error(f"Invalid JSON from {client_id[0]}:{client_id[1]}")
                    await websocket.send(
                        json.dumps(
                            {
                                "type": "error",
                                "error": error_msg,
                                "timestamp": time.time(),
                            }
                        )
                    )
                    continue

                if data["type"] in ["screen_frame", "video_frame", "image_input"]:
                    # Validate content exists
                    content = data.get("frame") or data.get("image")
                    if not content:
                        await websocket.send(
                            json.dumps(
                                {
                                    "type": "error",
                                    "error": "Missing image/frame content",
                                    "timestamp": time.time(),
                                }
                            )
                        )
                        continue

                    # Check content size (limit to 10MB for images)
                    if len(content) > 10 * 1024 * 1024:
                        await websocket.send(
                            json.dumps(
                                {
                                    "type": "error",
                                    "error": "Image/frame too large. Maximum size is 10MB.",
                                    "timestamp": time.time(),
                                }
                            )
                        )
                        continue

                    frame_data = {
                        "type": "image",
                        "timestamp": time.time(),
                        "content": content,
                        "prompt": data.get("prompt", "Describe what you see:"),
                        "source": data.get("source", "unknown"),
                    }

                    if not client_queue.full():
                        client_queue.put(frame_data)
                    else:
                        await websocket.send(
                            json.dumps({"type": "frame_dropped", "reason": "processing_queue_full"})
                        )

                elif data["type"] == "text_input":
                    # Validate content exists
                    if "content" not in data or not data["content"]:
                        await websocket.send(
                            json.dumps(
                                {
                                    "type": "error",
                                    "error": "Missing text content",
                                    "timestamp": time.time(),
                                }
                            )
                        )
                        continue

                    # Check content size (limit to 1MB)
                    if len(data["content"]) > 1024 * 1024:
                        await websocket.send(
                            json.dumps(
                                {
                                    "type": "error",
                                    "error": "Text content too large. Maximum size is 1MB.",
                                    "timestamp": time.time(),
                                }
                            )
                        )
                        continue

                    text_data = {
                        "type": "text",
                        "timestamp": time.time(),
                        "content": data["content"],
                        "context": data.get("context", ""),
                        "prompt": data["content"],
                    }

                    if not client_queue.full():
                        client_queue.put(text_data)
                    else:
                        await websocket.send(
                            json.dumps(
                                {
                                    "type": "error",
                                    "error": "Processing queue full, please retry",
                                    "timestamp": time.time(),
                                }
                            )
                        )

                elif data["type"] == "config":
                    await self._handle_config(data, websocket)

                elif data["type"] == "get_status":
                    await self._send_status(websocket)

        except websockets.exceptions.ConnectionClosed:
            if self.debug:
                logger.info(f"Client disconnected: {client_id}")
            else:
                log_info(f"Client disconnected: {client_id[0]}:{client_id[1]}")
        except asyncio.CancelledError:
            raise
        except Exception as e:
            error_msg = "Connection error occurred. Please reconnect."
            if self.debug:
                logger.error(f"Error handling client {client_id}: {e}", exc_info=True)
            else:
                log_error(f"Error with client {client_id[0]}:{client_id[1]}")
        finally:
            self.active_connections.discard(websocket)

            with self.clients_lock:
                if client_id in self.client_stop_events:
                    self.client_stop_events[client_id].set()

                if client_id in self.client_generators:
                    self.client_generators[client_id].clear()

            if "processing_thread" in locals():
                if client_id in self.client_stop_events:
                    self.client_stop_events[client_id].set()

                processing_thread.join(timeout=5.0)
                if processing_thread.is_alive():
                    if self.debug:
                        logger.warning(
                            f"Processing thread for {client_id} didn't stop cleanly after 5s (may still be draining queue)"
                        )

            with self.clients_lock:
                self.client_queues.pop(client_id, None)
                self.client_stop_events.pop(client_id, None)
                self.client_frame_counts.pop(client_id, None)
                self.client_generators.pop(client_id, None)

    def _process_frames(self, websocket, client_queue, stop_event, loop, client_id):
        """Process multimodal inputs in a separate thread"""
        messages_processed = 0
        drain_mode = False

        while not stop_event.is_set() or not drain_mode:
            try:
                timeout = 0.5 if not drain_mode else 0.1
                data = client_queue.get(timeout=timeout)

                if data["type"] == "image":
                    self._process_image(data, websocket, loop, client_id, stop_event)

                elif data["type"] == "text":
                    self._process_text(data, websocket, loop, client_id, stop_event)

                messages_processed += 1

                if self._message_processed_callback:
                    try:
                        self._message_processed_callback(client_id, data, messages_processed)
                    except Exception as e:
                        if self.debug:
                            logger.error(f"Error in message processed callback: {e}")

                if stop_event.is_set() and not drain_mode:
                    drain_mode = True
                    if self.debug:
                        logger.debug(
                            f"Entering drain mode for client {client_id}, queue size: {client_queue.qsize()}"
                        )

            except Empty:
                if stop_event.is_set() and not drain_mode:
                    drain_mode = True
                    if client_queue.empty():
                        break
                if drain_mode:
                    break
                continue
            except OSError as e:
                error_msg = "Network error"
                if self.debug:
                    logger.error(f"I/O error processing input for client {client_id}: {e}")
                self._safe_send(
                    websocket,
                    json.dumps({"type": "error", "error": error_msg, "timestamp": time.time()}),
                    loop,
                )
            except MemoryError as e:
                error_msg = "Out of memory"
                if self.debug:
                    logger.error(f"Memory error processing input for client {client_id}: {e}")
                self._safe_send(
                    websocket,
                    json.dumps({"type": "error", "error": error_msg, "timestamp": time.time()}),
                    loop,
                )
            except Exception as e:
                error_msg = "Processing thread encountered an error"
                if self.debug:
                    logger.error(
                        f"Error processing input for client {client_id}: {e}", exc_info=True
                    )
                else:
                    log_error(f"Processing error: {e}")
                self._safe_send(
                    websocket,
                    json.dumps({"type": "error", "error": error_msg, "timestamp": time.time()}),
                    loop,
                )

        remaining = 0
        try:
            while True:
                client_queue.get_nowait()
                remaining += 1
        except Empty:
            pass

        if self.debug:
            logger.debug(
                f"Processing thread stopped for client {client_id}. Messages processed: {messages_processed}, remaining: {remaining}"
            )

        if self._message_processed_callback and messages_processed > 0:
            try:
                self._message_processed_callback(client_id, None, -messages_processed)
            except Exception:
                pass

    def _safe_send(self, websocket, message, loop):
        """Safely send message to websocket, handling connection errors"""
        try:
            future = asyncio.run_coroutine_threadsafe(websocket.send(message), loop)
            future.result(timeout=1.0)
            return True
        except (
            websockets.exceptions.ConnectionClosed,
            websockets.exceptions.ConnectionClosedOK,
            websockets.exceptions.ConnectionClosedError,
            asyncio.TimeoutError,
            OSError,
            RuntimeError,
        ) as e:
            if self.debug and not isinstance(
                e, (websockets.exceptions.ConnectionClosedOK, asyncio.TimeoutError)
            ):
                error_msg = str(e)
                if "no close frame received or sent" not in error_msg:
                    logger.warning(f"Error sending message: {e}")
            return False

    def _stream_response(self, generator, websocket, loop, client_id, stop_event, data, input_type):
        """Common method for streaming token responses"""
        full_response = ""
        max_timeout = 60  # 60 seconds max for inference
        inference_start = time.time()

        try:
            for token in generator:
                # Check if we should stop
                if stop_event.is_set():
                    break

                # Check timeout
                if time.time() - inference_start > max_timeout:
                    raise TimeoutError(f"Inference exceeded {max_timeout}s timeout")

                full_response += token
                if not self._safe_send(
                    websocket,
                    json.dumps({"type": "token", "content": token, "timestamp": data["timestamp"]}),
                    loop,
                ):
                    break

            inference_time = time.time() - inference_start
            self._safe_send(
                websocket,
                json.dumps(
                    {
                        "type": "response_complete",
                        "full_text": full_response,
                        "timestamp": data["timestamp"],
                        "input_type": input_type,
                        "inference_time": inference_time,
                    }
                ),
                loop,
            )

            self._update_performance_metrics(inference_time)

            with self.clients_lock:
                if client_id in self.client_frame_counts:
                    self.client_frame_counts[client_id] += 1
                    frame_count = self.client_frame_counts[client_id]
                    if frame_count % 10 == 0 and self.debug:
                        logger.debug(
                            f"Client {client_id}: Processed {frame_count} inputs, last inference: {inference_time:.2f}s"
                        )

        finally:
            with self.clients_lock:
                if (
                    client_id in self.client_generators
                    and generator in self.client_generators[client_id]
                ):
                    self.client_generators[client_id].remove(generator)

    def _stream_response_with_cache(
        self, generator, websocket, loop, client_id, stop_event, data, input_type, cache_key
    ):
        """Stream response and cache the result"""
        full_response = ""
        max_timeout = 60  # 60 seconds max for inference
        inference_start = time.time()

        try:
            for token in generator:
                # Check if we should stop
                if stop_event.is_set():
                    break

                # Check timeout
                if time.time() - inference_start > max_timeout:
                    raise TimeoutError(f"Inference exceeded {max_timeout}s timeout")

                full_response += token
                if not self._safe_send(
                    websocket,
                    json.dumps({"type": "token", "content": token, "timestamp": data["timestamp"]}),
                    loop,
                ):
                    break

            inference_time = time.time() - inference_start
            self._safe_send(
                websocket,
                json.dumps(
                    {
                        "type": "response_complete",
                        "full_text": full_response,
                        "timestamp": data["timestamp"],
                        "input_type": input_type,
                        "inference_time": inference_time,
                    }
                ),
                loop,
            )

            if full_response:
                self._cache_response(cache_key, full_response)

            self._update_performance_metrics(inference_time)

            with self.clients_lock:
                if client_id in self.client_frame_counts:
                    self.client_frame_counts[client_id] += 1
                    frame_count = self.client_frame_counts[client_id]
                    if frame_count % 10 == 0 and self.debug:
                        logger.debug(
                            f"Client {client_id}: Processed {frame_count} inputs, last inference: {inference_time:.2f}s"
                        )

            return full_response

        finally:
            with self.clients_lock:
                if (
                    client_id in self.client_generators
                    and generator in self.client_generators[client_id]
                ):
                    self.client_generators[client_id].remove(generator)

    def _process_image(self, data, websocket, loop, client_id, stop_event):
        """Process image/video frame input"""
        self._ensure_model_loaded()
        try:
            # Decode base64 image
            image_data = data["content"]
            if "," in image_data:  # Handle data URL format
                image_bytes = base64.b64decode(image_data.split(",")[1])
            else:
                image_bytes = base64.b64decode(image_data)

            image = Image.open(io.BytesIO(image_bytes))

            # Resize if needed to save memory
            max_size = 768
            if max(image.size) > max_size:
                image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

            # Send start of response
            if not self._safe_send(
                websocket,
                json.dumps(
                    {
                        "type": "response_start",
                        "timestamp": data["timestamp"],
                        "input_type": "image",
                    }
                ),
                loop,
            ):
                return  # Connection closed, stop processing

            # Generate response with streaming
            # Use image-specific max tokens or fall back to config
            with self.config_lock:
                max_tokens = (
                    self.max_tokens_image if self.max_tokens_image else self.config["maxTokens"]
                )
                temperature = self.config["temperature"]
                top_p = self.config["topP"]

            # Use semaphore for concurrent inference
            with self.inference_semaphore:
                with self.clients_lock:
                    self.concurrent_inferences += 1
                # Get additional generation parameters
                with self.config_lock:
                    top_k = self.config["topK"]
                    repetition_penalty = self.config["repetitionPenalty"]
                    repetition_context_size = self.config["repetitionContextSize"]
                    seed = self.config["seed"]

                # Build generation kwargs
                gen_kwargs = {
                    "model": self.model,
                    "processor": self.processor,
                    "prompt": data["prompt"],
                    "image": image,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": top_k,
                    "repetition_penalty": repetition_penalty,
                    "repetition_context_size": repetition_context_size,
                    "stream": True,
                }

                # Add seed if specified
                if seed is not None:
                    gen_kwargs["seed"] = seed

                response_generator = generate(**gen_kwargs)

                # Track generator for cleanup
                with self.clients_lock:
                    if client_id in self.client_generators:
                        self.client_generators[client_id].append(response_generator)

            try:
                # Use common streaming method
                self._stream_response(
                    response_generator, websocket, loop, client_id, stop_event, data, "image"
                )
            finally:
                with self.clients_lock:
                    self.concurrent_inferences -= 1

        except ValueError as e:
            error_msg = "Invalid image data. Please ensure the image is properly encoded."
            if self.debug:
                logger.error(f"Image validation error: {e}")
            self._safe_send(
                websocket,
                json.dumps({"type": "error", "error": error_msg, "timestamp": data["timestamp"]}),
                loop,
            )
        except MemoryError as e:
            error_msg = "Out of memory processing image. Try a smaller image or lower resolution."
            if self.debug:
                logger.error(f"Memory error processing image: {e}")
            self._safe_send(
                websocket,
                json.dumps({"type": "error", "error": error_msg, "timestamp": data["timestamp"]}),
                loop,
            )
        except Exception as e:
            error_msg = "Failed to process image. Please check the image format and try again."
            if self.debug:
                logger.error(f"Error processing image: {e}", exc_info=True)
            self._safe_send(
                websocket,
                json.dumps({"type": "error", "error": error_msg, "timestamp": data["timestamp"]}),
                loop,
            )

    def _process_text(self, data, websocket, loop, client_id, stop_event):
        """Process text-only input"""
        self._ensure_model_loaded()
        try:
            # Send start of response
            if not self._safe_send(
                websocket,
                json.dumps(
                    {"type": "response_start", "timestamp": data["timestamp"], "input_type": "text"}
                ),
                loop,
            ):
                return  # Connection closed, stop processing

            # Build prompt with context if provided
            prompt = data["prompt"]
            if data.get("context"):
                prompt = f"Context: {data['context']}\n\nUser: {prompt}"

            # Get config with lock
            with self.config_lock:
                max_tokens = self.config["maxTokens"]
                temperature = self.config["temperature"]
                top_p = self.config["topP"]
                top_k = self.config["topK"]
                repetition_penalty = self.config["repetitionPenalty"]
                repetition_context_size = self.config["repetitionContextSize"]
                seed = self.config["seed"]

            # Check cache first
            cache_key = self._get_cache_key(prompt, self.config)
            cached_response = self._get_cached_response(cache_key)

            if cached_response:
                # Send cached response
                if not self._safe_send(
                    websocket,
                    json.dumps(
                        {
                            "type": "token",
                            "content": cached_response,
                            "timestamp": data["timestamp"],
                        }
                    ),
                    loop,
                ):
                    return

                self._safe_send(
                    websocket,
                    json.dumps(
                        {
                            "type": "response_complete",
                            "full_text": cached_response,
                            "timestamp": data["timestamp"],
                            "input_type": "text",
                            "inference_time": 0.0,
                            "cached": True,
                        }
                    ),
                    loop,
                )
                return

            # Generate response with streaming
            # Use semaphore for concurrent inference
            with self.inference_semaphore:
                with self.clients_lock:
                    self.concurrent_inferences += 1
                try:
                    # Build generation kwargs
                    gen_kwargs = {
                        "model": self.model,
                        "processor": self.processor,
                        "prompt": prompt,
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "top_p": top_p,
                        "top_k": top_k,
                        "repetition_penalty": repetition_penalty,
                        "repetition_context_size": repetition_context_size,
                        "stream": True,
                    }

                    # Add seed if specified
                    if seed is not None:
                        gen_kwargs["seed"] = seed

                    # Try vision model API first
                    response_generator = generate(**gen_kwargs)
                except (AttributeError, TypeError):
                    # If vision API fails, try text-only API
                    # This allows compatibility with non-multimodal models
                    if text_generate is not None:
                        # Build text generation kwargs
                        text_kwargs = {
                            "model": self.model,
                            "tokenizer": self.processor,  # text_generate uses processor as tokenizer
                            "prompt": prompt,
                            "max_tokens": max_tokens,
                            "temperature": temperature,
                            "top_p": top_p,
                            "top_k": top_k,
                            "repetition_penalty": repetition_penalty,
                            "repetition_context_size": repetition_context_size,
                            "stream": True,
                        }

                        # Add seed if specified
                        if seed is not None:
                            text_kwargs["seed"] = seed

                        response_generator = text_generate(**text_kwargs)
                    else:
                        raise TextGenerationError(
                            "Model does not support text generation. Please load a text-capable model."
                        ) from None

                # Track generator for cleanup
                with self.clients_lock:
                    if client_id in self.client_generators:
                        self.client_generators[client_id].append(response_generator)

            try:
                # Use common streaming method
                full_response = self._stream_response_with_cache(
                    response_generator,
                    websocket,
                    loop,
                    client_id,
                    stop_event,
                    data,
                    "text",
                    cache_key,
                )
            finally:
                with self.clients_lock:
                    self.concurrent_inferences -= 1

        except AttributeError as e:
            error_msg = "Text generation model not available. Please load a text-capable model."
            if self.debug:
                logger.error(f"Model API error: {e}")
            self._safe_send(
                websocket,
                json.dumps({"type": "error", "error": error_msg, "timestamp": data["timestamp"]}),
                loop,
            )
        except ValueError as e:
            error_msg = "Invalid text input. Please check your message format."
            if self.debug:
                logger.error(f"Text validation error: {e}")
            self._safe_send(
                websocket,
                json.dumps({"type": "error", "error": error_msg, "timestamp": data["timestamp"]}),
                loop,
            )
        except Exception as e:
            error_msg = "Failed to process text. Please try again."
            if self.debug:
                logger.error(f"Error processing text: {e}", exc_info=True)
            self._safe_send(
                websocket,
                json.dumps({"type": "error", "error": error_msg, "timestamp": data["timestamp"]}),
                loop,
            )

    async def _handle_config(self, config_update, websocket):
        """Handle configuration updates supporting both MLX-native and OpenAI formats

        MLX-native parameters:
        - maxTokens: Maximum tokens to generate
        - temperature: Randomness control [0.0, 1.0]
        - topP: Nucleus sampling threshold
        - topK: Top-k sampling
        - repetitionPenalty: Token repetition penalty [1.0, 2.0]
        - repetitionContextSize: Context window for repetition
        - seed: Random seed for reproducibility

        OpenAI-compatible parameters (converted to MLX):
        - maxOutputTokens → maxTokens
        - presencePenalty → repetitionPenalty
        - frequencyPenalty → repetitionPenalty
        - candidateCount: Number of responses (MLX only supports 1)
        """
        updated = {}
        conversions = []  # Track parameter conversions for logging

        # Handle candidateCount (both formats)
        if "candidateCount" in config_update:
            try:
                value = int(config_update["candidateCount"])
                if value != 1:
                    conversions.append("candidateCount > 1 not supported, using 1")
                    value = 1
                with self.config_lock:
                    self.config["candidateCount"] = value
                updated["candidateCount"] = value
            except (ValueError, TypeError):
                if self.debug:
                    logger.warning(f"Invalid candidateCount: {config_update['candidateCount']}")

        # Handle maxTokens / maxOutputTokens
        if "maxTokens" in config_update:
            try:
                value = int(config_update["maxTokens"])
                if value > 0:
                    with self.config_lock:
                        self.config["maxTokens"] = value
                    updated["maxTokens"] = value
            except (ValueError, TypeError):
                if self.debug:
                    logger.warning(f"Invalid maxTokens: {config_update['maxTokens']}")

        elif "maxOutputTokens" in config_update:  # OpenAI style
            try:
                value = int(config_update["maxOutputTokens"])
                if value > 0:
                    conversions.append(f"maxOutputTokens → maxTokens ({value})")
                    with self.config_lock:
                        self.config["maxTokens"] = value
                    updated["maxTokens"] = value
            except (ValueError, TypeError):
                if self.debug:
                    logger.warning(f"Invalid maxOutputTokens: {config_update['maxOutputTokens']}")

        # Handle temperature
        if "temperature" in config_update:
            try:
                value = float(config_update["temperature"])
                # Clamp to MLX range [0.0, 1.0]
                temp = max(0.0, min(1.0, value))
                if value > 1.0:
                    conversions.append(f"temperature clamped from {value} to 1.0")
                with self.config_lock:
                    self.config["temperature"] = temp
                updated["temperature"] = temp
            except (ValueError, TypeError):
                if self.debug:
                    logger.warning(f"Invalid temperature: {config_update['temperature']}")

        # Handle topP
        if "topP" in config_update:
            try:
                value = float(config_update["topP"])
                top_p = max(0.0, min(1.0, value))
                with self.config_lock:
                    self.config["topP"] = top_p
                updated["topP"] = top_p
            except (ValueError, TypeError):
                if self.debug:
                    logger.warning(f"Invalid topP: {config_update['topP']}")

        # Handle topK
        if "topK" in config_update:
            try:
                value = int(config_update["topK"])
                top_k = max(1, value)
                with self.config_lock:
                    self.config["topK"] = top_k
                updated["topK"] = top_k
            except (ValueError, TypeError):
                if self.debug:
                    logger.warning(f"Invalid topK: {config_update['topK']}")

        # Handle repetitionPenalty (MLX-native)
        if "repetitionPenalty" in config_update:
            try:
                value = float(config_update["repetitionPenalty"])
                # MLX range is typically 1.0-2.0
                penalty = max(1.0, min(2.0, value))
                with self.config_lock:
                    self.config["repetitionPenalty"] = penalty
                updated["repetitionPenalty"] = penalty
            except (ValueError, TypeError):
                if self.debug:
                    logger.warning(
                        f"Invalid repetitionPenalty: {config_update['repetitionPenalty']}"
                    )

        # Handle OpenAI-style penalties (convert to repetitionPenalty)
        elif "presencePenalty" in config_update or "frequencyPenalty" in config_update:
            try:
                presence = float(config_update.get("presencePenalty", 0.0))
                frequency = float(config_update.get("frequencyPenalty", 0.0))

                # Convert OpenAI penalties to MLX repetitionPenalty
                # OpenAI range [-2.0, 2.0] -> MLX range [1.0, 2.0]
                penalty_strength = max(abs(presence), abs(frequency))
                repetition_penalty = 1.0 + (penalty_strength * 0.25)
                repetition_penalty = max(1.0, min(2.0, repetition_penalty))

                conversions.append(
                    f"presencePenalty({presence})/frequencyPenalty({frequency}) → repetitionPenalty({repetition_penalty:.2f})"
                )

                with self.config_lock:
                    self.config["repetitionPenalty"] = repetition_penalty
                updated["repetitionPenalty"] = repetition_penalty
            except (ValueError, TypeError) as e:
                if self.debug:
                    logger.warning(f"Invalid penalty values: {e}")

        # Handle repetitionContextSize (MLX-native)
        if "repetitionContextSize" in config_update:
            try:
                value = int(config_update["repetitionContextSize"])
                ctx_size = max(1, value)
                with self.config_lock:
                    self.config["repetitionContextSize"] = ctx_size
                updated["repetitionContextSize"] = ctx_size
            except (ValueError, TypeError):
                if self.debug:
                    logger.warning(
                        f"Invalid repetitionContextSize: {config_update['repetitionContextSize']}"
                    )

        # Handle seed
        if "seed" in config_update:
            try:
                value = config_update["seed"]
                if value is None:
                    seed = None
                else:
                    seed = int(value)
                with self.config_lock:
                    self.config["seed"] = seed
                updated["seed"] = seed
            except (ValueError, TypeError):
                if self.debug:
                    logger.warning(f"Invalid seed: {config_update['seed']}")

        # Handle responseModalities
        if "responseModalities" in config_update:
            try:
                if isinstance(config_update["responseModalities"], list):
                    valid_modalities = ["TEXT", "IMAGE", "AUDIO"]
                    modalities = [
                        m for m in config_update["responseModalities"] if m in valid_modalities
                    ]

                    if any(m != "TEXT" for m in modalities):
                        conversions.append("Only TEXT modality is currently supported")
                        modalities = ["TEXT"]

                    with self.config_lock:
                        self.config["responseModalities"] = modalities
                    updated["responseModalities"] = modalities
            except Exception as e:
                if self.debug:
                    logger.error(f"Invalid responseModalities: {e}")

        # Special handling for convenience max_tokens_image
        if "max_tokens_image" in config_update:
            try:
                value = int(config_update["max_tokens_image"])
                if value > 0:
                    with self.config_lock:
                        self.max_tokens_image = value
                    updated["max_tokens_image"] = value
                else:
                    if self.debug:
                        logger.warning(
                            f"Invalid max_tokens_image value: {value} (must be positive)"
                        )
            except (ValueError, TypeError):
                if self.debug:
                    logger.warning(
                        f"Invalid max_tokens_image value: {config_update['max_tokens_image']}"
                    )

        # Log any parameter conversions
        if conversions:
            conversion_msg = "Config conversions: " + "; ".join(conversions)
            if self.debug:
                logger.info(conversion_msg)
            else:
                log_info(conversion_msg)

        # Send confirmation with updated fields and full current config
        with self.config_lock:
            current_config = self.config.copy()

        response = {
            "type": "config_updated",
            "updated_fields": updated,
            "current_config": current_config,
        }

        # Include conversions in response if any occurred
        if conversions:
            response["conversions"] = conversions

        await websocket.send(json.dumps(response))

    async def _send_status(self, websocket):
        """Send server status including resource information."""
        try:
            stats = self.resource_monitor.get_resource_stats()

            # Add model info
            stats["model"] = {"name": self.model_name, "loaded": self.model is not None}

            # Add server config
            with self.config_lock:
                stats["config"] = self.config.copy()

            await websocket.send(
                json.dumps({"type": "status", "stats": stats, "timestamp": time.time()})
            )
        except Exception as e:
            logger.error(f"Error sending status: {e}")
            await websocket.send(
                json.dumps(
                    {
                        "type": "error",
                        "error": "Failed to get server status",
                        "timestamp": time.time(),
                    }
                )
            )

    async def shutdown(self):
        """Gracefully shutdown the server"""
        if not self.debug:
            # Use red color for shutdown messages
            if RICH_AVAILABLE:
                console.print("\n[red]Shutting down server...[/red]")
            else:
                sys.stdout.write(
                    f"\n{ANSI_COLORS['red']}Shutting down server...{ANSI_COLORS['reset']}\n"
                )
                sys.stdout.flush()
        else:
            logger.info("Shutting down server...")

        # Stop accepting new connections
        self.shutdown_event.set()

        # Stop resource monitoring
        if hasattr(self, "resource_monitor"):
            self.resource_monitor.stop()

        # Close all active WebSocket connections
        if self.active_connections:
            if not self.debug:
                log_warning(f"Closing {len(self.active_connections)} active connections...")
            else:
                logger.info(f"Closing {len(self.active_connections)} active connections...")
            close_tasks = [ws.close() for ws in self.active_connections]
            await asyncio.gather(*close_tasks, return_exceptions=True)

        # Signal all client threads to stop
        with self.clients_lock:
            for stop_event in self.client_stop_events.values():
                stop_event.set()

            # Clear generators to signal no more processing
            for client_id in self.client_generators:
                self.client_generators[client_id].clear()

        # Wait for threads to finish with a longer timeout
        # This is especially important during tests with many concurrent clients
        max_wait_time = 5.0  # Maximum time to wait for threads
        wait_interval = 0.1  # Check interval
        elapsed_time = 0.0

        while elapsed_time < max_wait_time:
            # Check if any threads are still running
            active_threads = []
            for thread in threading.enumerate():
                if thread.name.startswith("Thread-") and thread.is_alive():
                    # Check if it's one of our processing threads
                    if (
                        hasattr(thread, "_target")
                        and thread._target
                        and thread._target.__name__ == "_process_frames"
                    ):
                        active_threads.append(thread)

            if not active_threads:
                break  # All threads finished

            await asyncio.sleep(wait_interval)
            elapsed_time += wait_interval

        if active_threads and self.debug:
            logger.warning(f"Timed out waiting for {len(active_threads)} threads to finish")

        # Close the server
        if self.server:
            # Handle both sync and async close methods (for mocking compatibility)
            close_result = self.server.close()
            if asyncio.iscoroutine(close_result):
                await close_result
            await self.server.wait_closed()

        # Clean up MLX resources
        if self.model is not None:
            if not self.debug:
                log_warning("Cleaning up MLX model...")
            else:
                logger.info("Cleaning up MLX model...")
            # MLX doesn't have explicit cleanup, but we can clear references
            self.model = None
            self.processor = None
            # Force garbage collection to free MLX memory
            import gc

            gc.collect()
            if mx is not None:
                mx.clear_cache()

        if not self.debug:
            log_success("Server shutdown complete\n")
        else:
            logger.info("Server shutdown complete")

    def _get_network_addresses(self):
        """Get all local network addresses"""
        addresses = []

        # Try primary method
        try:
            # Get hostname
            hostname = socket.gethostname()
            # Get all IPs associated with hostname
            host_ips = socket.gethostbyname_ex(hostname)[2]
            # Filter out loopback addresses
            for ip in host_ips:
                if not ip.startswith("127."):
                    addresses.append(ip)
        except OSError:
            pass  # Try fallback method

        # Alternative method using socket connection
        if not addresses:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                    # Connect to a public DNS server
                    s.connect(("8.8.8.8", 80))
                    addresses.append(s.getsockname()[0])
            except OSError as e:
                if self.debug:
                    logger.warning(f"Could not determine network addresses: {e}")
                # Return empty list, main address will still work

        return addresses

    def _register_cleanup(self):
        """Register cleanup handlers to prevent orphaned processes."""
        if self._cleanup_registered:
            return

        # Register atexit handler
        import atexit

        atexit.register(self._emergency_cleanup)

        # Enhanced signal handling
        def signal_handler(signum, frame):
            # Set shutdown flag immediately
            self.shutting_down = True
            # Create task in the event loop
            try:
                loop = asyncio.get_running_loop()
                if loop and loop.is_running():
                    loop.call_soon_threadsafe(lambda: asyncio.create_task(self.shutdown()))
                else:
                    # Fallback: direct exit
                    self._emergency_cleanup()
                    sys.exit(0)
            except RuntimeError:
                # No event loop running
                self._emergency_cleanup()
                sys.exit(0)

        # Try to register signal handlers
        try:
            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)
            if hasattr(signal, "SIGHUP"):
                signal.signal(signal.SIGHUP, signal_handler)
        except Exception as e:
            if self.debug:
                logger.warning(f"Could not register signal handlers: {e}")

        self._cleanup_registered = True

    def _emergency_cleanup(self):
        """Emergency cleanup for unexpected exits."""
        try:
            # Unregister from process registry
            if hasattr(self, "actual_port") and self.actual_port:
                registry = get_registry()
                registry.unregister_process(os.getpid())

            if hasattr(self, "resource_monitor") and self.resource_monitor:
                self.resource_monitor.stop()

            # Note: server.close() is async and can't be called from emergency cleanup
            # The process will exit anyway, so explicit close isn't necessary

            # Clear model from memory
            self.model = None
            self.processor = None

            if self.debug:
                logger.info("Emergency cleanup completed")
        except Exception:
            pass  # Best effort in emergency

    async def _find_available_port(self, start_port: int, max_tries: int = 100) -> int:
        """Find next available port starting from start_port."""
        import socket

        # Get process registry to check for MLX processes
        registry = get_registry()
        mlx_ports = registry.find_mlx_process_ports()

        for offset in range(max_tries):
            test_port = start_port + offset

            # Check if port is used by an MLX process
            if test_port in mlx_ports:
                existing_process = registry.find_process_on_port(test_port)
                if existing_process:
                    log_warning(
                        f"Port {test_port} is already used by another MLX instance (PID: {existing_process.pid})"
                    )
                continue

            # Quick socket test
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                try:
                    sock.bind((self.host, test_port))
                    sock.close()

                    # Double-check with asyncio
                    try:
                        test_server = await asyncio.start_server(
                            lambda r, w: None, self.host, test_port
                        )
                        test_server.close()
                        await test_server.wait_closed()

                        if test_port != start_port:
                            log_info(f"Port {start_port} busy, using port {test_port}")

                        return test_port
                    except OSError:
                        continue

                except OSError:
                    continue

        raise NetworkError(
            f"No available ports found in range {start_port}-{start_port + max_tries}.\n"
            f"Try using port 0 to let the OS assign a port: mlx serve --port 0"
        )

    async def start_server(self):
        """Start the WebSocket server with graceful shutdown"""
        # Signal handlers already set up in _register_cleanup()

        # React-style startup messages
        if RICH_AVAILABLE:
            console.print("\n" + "[cyan]=" * 60 + "[/cyan]")
            console.print("[bold_white]MLX WebSocket Streaming Server v0.1.0[/bold_white]")
            console.print("[cyan]=" * 60 + "[/cyan]\n")

            console.print(f"[cream]Model:[/cream] [bold_white]{self.model_name}[/bold_white]")
            (
                console.print(
                    f"[cream]Memory:[/cream] [white]{mx.get_active_memory() / 1024**3:.2f} GB[/white]\n"
                )
                if mx
                else console.print("[cream]Memory:[/cream] [red]Not available[/red]\n")
            )
        else:
            # Use standardized colors
            sys.stdout.write(f"\n{ANSI_COLORS['purple']}{'=' * 60}{ANSI_COLORS['reset']}\n")
            sys.stdout.write(
                f"{ANSI_COLORS['cream']}MLX WebSocket Streaming Server v0.1.0{ANSI_COLORS['reset']}\n"
            )
            sys.stdout.write(f"{ANSI_COLORS['purple']}{'=' * 60}{ANSI_COLORS['reset']}\n\n")

            sys.stdout.write(
                f"{ANSI_COLORS['cream']}Model:{ANSI_COLORS['reset']} {ANSI_COLORS['purple']}{self.model_name}{ANSI_COLORS['reset']}\n"
            )
            if mx:
                sys.stdout.write(
                    f"{ANSI_COLORS['cream']}Memory:{ANSI_COLORS['reset']} {ANSI_COLORS['purple']}{mx.get_active_memory() / 1024**3:.2f}{ANSI_COLORS['reset']} GB\n\n"
                )
            else:
                sys.stdout.write(
                    f"{ANSI_COLORS['cream']}Memory:{ANSI_COLORS['reset']} {ANSI_COLORS['purple']}Not available{ANSI_COLORS['reset']}\n\n"
                )
            sys.stdout.flush()

        try:
            # Check for existing MLX processes before starting
            registry = get_registry()
            existing_processes = registry.get_all_processes()

            # Check if requested port is already used by an MLX process
            if self.requested_port != 0:
                for proc in existing_processes:
                    if proc.port == self.requested_port:
                        raise NetworkError(
                            f"Port {self.requested_port} is already in use by another MLX instance:\n"
                            f"  PID: {proc.pid}\n"
                            f"  Model: {proc.model or 'unknown'}\n"
                            f"  Type: {'daemon-managed' if proc.is_daemon else 'standalone'}\n\n"
                            f"Options:\n"
                            f"  1. Stop the existing instance: mlx bg stop\n"
                            f"  2. Use auto port discovery: mlx serve (without --port)\n"
                            f"  3. Let OS assign a port: mlx serve --port 0"
                        )

            # Find available port if needed
            if self.auto_port and self.requested_port != 0:
                self.actual_port = await self._find_available_port(self.requested_port)
            else:
                self.actual_port = self.requested_port

            # Configure websockets loggers to handle errors gracefully
            ws_filter = WebSocketHandshakeFilter(self)

            # Apply filter to all websockets-related loggers
            for logger_name in [
                "websockets",
                "websockets.server",
                "websockets.protocol",
                "websockets.asyncio",
            ]:
                logger = logging.getLogger(logger_name)
                logger.addFilter(ws_filter)
                # Also set a higher level to reduce verbosity
                if not self.debug:
                    logger.setLevel(logging.WARNING)

            # Create custom logger for websockets
            ws_logger = CustomWebSocketLogger(logging.getLogger("websockets.server"), self)

            # Define process_request to handle non-WebSocket connections gracefully
            async def process_request(connection, request):
                """Handle HTTP requests that aren't WebSocket upgrades."""
                # This is called for every connection attempt
                # Return None to continue with WebSocket handshake
                # Return a Response to reject the connection
                return None

            self.server = await websockets.serve(
                self.handle_client,
                self.host,
                self.actual_port,
                max_size=20 * 1024 * 1024,  # 20MB max message size
                process_request=process_request,
                logger=ws_logger if self.debug else None,  # Use custom logger in debug mode
            )

            # If port was 0, get actual assigned port
            if self.actual_port == 0:
                self.actual_port = self.server.sockets[0].getsockname()[1]

            # Register this process with the registry
            registry.register_process(
                ProcessInfo(
                    pid=os.getpid(),
                    port=self.actual_port,
                    model=self.model_name,
                    start_time=time.time(),
                    is_daemon=False,  # This is a standalone server
                )
            )

            # Start resource monitoring
            self.resource_monitor.start()

            # Display connection URLs
            log_info("Server running at:")
            log_info(f"  Local:            ws://localhost:{self.actual_port}")

            # Get network addresses
            network_addresses = self._get_network_addresses()
            if network_addresses:
                for ip in network_addresses:
                    log_info(f"  On Your Network:  ws://{ip}:{self.actual_port}")

            # Add a blank line without arrow
            print()
            log_info("Note: To connect from other devices, ensure they are on the same network.")
            log_info("Press Ctrl+C to stop the server.")
            print()  # Print blank line without arrow

            # Wait until shutdown is requested
            await self.shutdown_event.wait()

        except OSError as e:
            if e.errno == 48:  # Address in use
                raise NetworkError(
                    f"Port {self.actual_port} is still in use after discovery.\n"
                    f"This might be a race condition. Try again or specify --port 0"
                ) from e
            raise
        except asyncio.CancelledError:
            if self.debug:
                logger.info("Server task cancelled")
        finally:
            # Unregister from process registry
            registry = get_registry()
            registry.unregister_process(os.getpid())

            # Stop resource monitoring
            if hasattr(self, "resource_monitor"):
                self.resource_monitor.stop()

            # Ensure cleanup happens even on unexpected exit
            if not self.shutdown_event.is_set():
                await self.shutdown()


def main():
    """Main entry point for the server."""
    import argparse

    parser = argparse.ArgumentParser(
        description="MLX WebSocket Streaming Server - Real-time multimodal AI inference server for Apple Silicon Macs. "
        "Supports streaming video frames, images, and text chat with local MLX models.",
        epilog="Example: python -m mlx_websockets --model mlx-community/gemma-3-4b-it-4bit --port 8765",
    )
    parser.add_argument(
        "--model",
        default="mlx-community/gemma-3-4b-it-4bit",
        help="MLX model name from HuggingFace",
    )
    parser.add_argument("--port", type=int, default=8765, help="WebSocket server port")
    parser.add_argument(
        "--auto-port",
        action="store_true",
        help="Enable automatic port discovery if the preferred port is busy",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind to (use 'localhost' for local-only access)",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Enable trusting remote code for tokenizer",
    )
    parser.add_argument(
        "--tokenizer-config",
        type=str,
        help="Tokenizer config.json file",
    )
    parser.add_argument(
        "--chat-template",
        type=str,
        help="Chat template or template name",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for generation",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    parser.add_argument(
        "--show-warnings",
        action="store_true",
        help="Show all warnings (including harmless ones)",
    )

    args = parser.parse_args()

    # Configure logging based on debug flag
    log_level = "DEBUG" if args.debug else "INFO"
    setup_logging(level=log_level, use_rich=not args.debug)

    # Re-enable warnings if requested
    if args.show_warnings:
        warnings.resetwarnings()
        if RICH_AVAILABLE:
            console.print(
                "[purple italic]Note: Showing all warnings. These are typically harmless.[/purple italic]"
            )

    # Import dependencies with loading indicator
    _import_dependencies(debug=args.debug)

    # Start server
    server = MLXStreamingServer(
        model_name=args.model,
        port=args.port,
        host=args.host,
        debug=args.debug,
        auto_port=args.auto_port,
    )
    asyncio.run(server.start_server())


if __name__ == "__main__":
    main()
