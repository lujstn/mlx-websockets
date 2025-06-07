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
import threading
import time
import warnings
from queue import Empty, Queue
from threading import Lock, RLock

# Suppress specific warnings before imports
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

# Suppress specific loggers
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning:multiprocessing.resource_tracker"

# Configure logging - will be set up after parsing args
logger = logging.getLogger(__name__)

# Rich console support for better output (optional)
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
    from rich.text import Text

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


# Import heavy dependencies with loading indicator
def _import_dependencies(debug=False):
    """Import MLX and other heavy dependencies with loading indicator"""
    global mx, websockets, generate, load, Image, text_generate

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

            import websockets as ws

            websockets = ws
            progress.update(task, description="Loading MLX models...")

            from mlx_vlm import generate as gen
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
        # Normal imports without progress bar
        import mlx.core as mx
        import websockets as ws

        websockets = ws
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


class MLXStreamingServer:
    def __init__(
        self, model_name="mlx-community/gemma-3-4b-it-4bit", port=8765, host="0.0.0.0", debug=False
    ):
        """
        Initialize the MLX WebSocket Streaming Server

        Args:
            model_name: MLX model from HuggingFace (Gemma 3 4bit model recommended)
            port: WebSocket server port (default 8765)
            host: Host to bind to (default 0.0.0.0 for all interfaces)
            debug: Enable debug logging (default False)
        """
        self.model_name = model_name
        self.port = port
        self.host = host
        self.debug = debug
        self.model = None
        self.processor = None

        # Thread safety
        self.clients_lock = RLock()  # Protects client dictionaries
        self.config_lock = RLock()  # Protects configuration
        self.model_lock = Lock()  # Protects model inference

        # Client state management (protected by clients_lock)
        self.client_queues = {}
        self.client_stop_events = {}  # Stop events for thread termination
        self.client_frame_counts = {}  # Per-client frame counting
        self.client_generators = {}  # Track active generators for cleanup

        # Default configuration following formal schema (protected by config_lock)
        self.config = {
            "candidateCount": 1,  # MLX models typically generate 1 response
            "maxOutputTokens": 200,  # Default max tokens
            "temperature": 0.7,
            "topP": 1.0,  # Default: consider all tokens
            "topK": 50,  # Default: consider top 50 tokens
            "presencePenalty": 0.0,  # No presence penalty by default
            "frequencyPenalty": 0.0,  # No frequency penalty by default
            "responseModalities": ["TEXT"],  # Default to text only
        }

        # Separate max tokens for different input types (convenience)
        # Protected by config_lock since it's configuration
        self.max_tokens_image = 100  # Images typically need fewer tokens

        # Shutdown handling
        self.shutdown_event = asyncio.Event()
        self.active_connections = set()  # Track active WebSocket connections
        self.server = None  # Will hold the WebSocket server

        if not debug:
            if RICH_AVAILABLE:
                console.print(f"[cream]Loading model:[/cream] [yellow]{model_name}[/yellow]")
            else:
                # ANSI codes: white=#fffbeb approx, yellow=#ffd230
                print(f"\033[97mLoading model:\033[0m \033[93m{model_name}\033[0m...")
        else:
            logger.info(f"Loading model: {model_name}")
        self._load_model()

    def _load_model(self):
        """Load the MLX model and processor"""
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
                if RICH_AVAILABLE:
                    console.print("[green]✓[/green] [cream]Model loaded successfully![/cream]")
                else:
                    print("\033[92m✓\033[0m \033[97mModel loaded successfully!\033[0m")
            else:
                logger.info("Model loaded successfully!")
                logger.info(f"Memory usage: {mx.get_active_memory() / 1024**3:.2f} GB")
        except Exception as e:
            if not self.debug:
                if RICH_AVAILABLE:
                    console.print(f"[red]✗[/red] [cream]Error loading model:[/cream] {e}")
                else:
                    print(f"\033[91m✗\033[0m \033[97mError loading model:\033[0m {e}")
            else:
                logger.error(f"Error loading model: {e}")
            raise

    async def handle_client(self, websocket, path):
        """Handle WebSocket client connections"""
        client_id = websocket.remote_address
        if self.debug:
            logger.info(f"Client connected: {client_id}")
        else:
            if RICH_AVAILABLE:
                console.print(
                    f"[green]→[/green] [pink]Client connected:[/pink] [cyan]{client_id[0]}:{client_id[1]}[/cyan]"
                )
            else:
                # ANSI: green=92, pink=95, cyan=96
                print(
                    f"\033[92m→\033[0m \033[95mClient connected:\033[0m \033[96m{client_id[0]}:{client_id[1]}\033[0m"
                )

        # Track this connection for shutdown
        self.active_connections.add(websocket)

        # Create client state
        client_queue = Queue(maxsize=10)
        stop_event = threading.Event()

        with self.clients_lock:
            self.client_queues[client_id] = client_queue
            self.client_stop_events[client_id] = stop_event
            self.client_frame_counts[client_id] = 0
            self.client_generators[client_id] = []

        # Get the current event loop
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.get_event_loop()

        # Start processing thread for this client
        processing_thread = threading.Thread(
            target=self._process_frames,
            args=(websocket, client_queue, stop_event, loop, client_id),
            daemon=False,  # Non-daemon so we can wait for it to finish
        )
        processing_thread.start()

        try:
            async for message in websocket:
                data = json.loads(message)

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

                    # Add frame/image to processing queue
                    frame_data = {
                        "type": "image",
                        "timestamp": time.time(),
                        "content": content,
                        "prompt": data.get("prompt", "Describe what you see:"),
                        "source": data.get("source", "unknown"),
                    }

                    # Skip frame if queue is full (drop frames to maintain real-time)
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

                    # Handle text input directly (no image)
                    text_data = {
                        "type": "text",
                        "timestamp": time.time(),
                        "content": data["content"],
                        "context": data.get("context", ""),
                        "prompt": data["content"],
                    }

                    # Check queue capacity
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
                    # Handle configuration updates
                    await self._handle_config(data, websocket)

        except websockets.exceptions.ConnectionClosed:
            if self.debug:
                logger.info(f"Client disconnected: {client_id}")
            else:
                if RICH_AVAILABLE:
                    console.print(
                        f"[yellow]←[/yellow] [pink]Client disconnected:[/pink] [cyan]{client_id[0]}:{client_id[1]}[/cyan]"
                    )
                else:
                    # ANSI: yellow=93
                    print(
                        f"\033[93m←\033[0m \033[95mClient disconnected:\033[0m \033[96m{client_id[0]}:{client_id[1]}\033[0m"
                    )
        except Exception as e:
            if self.debug:
                logger.error(f"Error handling client {client_id}: {e}", exc_info=True)
            else:
                if RICH_AVAILABLE:
                    console.print(
                        f"[red]✗[/red] [cream]Error with client[/cream] [cyan]{client_id[0]}:{client_id[1]}[/cyan]: {e}"
                    )
                else:
                    print(
                        f"\033[91m✗\033[0m \033[97mError with client\033[0m \033[96m{client_id[0]}:{client_id[1]}\033[0m: {e}"
                    )
        finally:
            # Remove from active connections
            self.active_connections.discard(websocket)

            # Signal thread to stop and clean up generators
            with self.clients_lock:
                if client_id in self.client_stop_events:
                    self.client_stop_events[client_id].set()

                # Cancel any active generators
                if client_id in self.client_generators:
                    # Clear the list to signal no more processing needed
                    self.client_generators[client_id].clear()

            # Wait for the processing thread to finish (up to 2 seconds)
            if "processing_thread" in locals():
                processing_thread.join(timeout=2.0)
                if processing_thread.is_alive():
                    if self.debug:
                        logger.warning(f"Processing thread for {client_id} didn't stop cleanly")

            # Clean up client state with lock
            with self.clients_lock:
                self.client_queues.pop(client_id, None)
                self.client_stop_events.pop(client_id, None)
                self.client_frame_counts.pop(client_id, None)
                self.client_generators.pop(client_id, None)

    def _process_frames(self, websocket, client_queue, stop_event, loop, client_id):
        """Process multimodal inputs in a separate thread"""
        while not stop_event.is_set():
            try:
                # Check queue with timeout to allow periodic stop checks
                data = client_queue.get(timeout=0.5)

                if data["type"] == "image":
                    # Process image/video frame
                    self._process_image(data, websocket, loop, client_id, stop_event)

                elif data["type"] == "text":
                    # Process text-only input
                    self._process_text(data, websocket, loop, client_id, stop_event)

            except Empty:
                # Timeout is expected - check if we should stop
                continue
            except Exception as e:
                if self.debug:
                    logger.error(f"Error processing input for client {client_id}: {e}")
                else:
                    if RICH_AVAILABLE:
                        console.print(f"[red]✗[/red] [cream]Processing error:[/cream] {e}")
                    else:
                        print(f"\033[91m✗\033[0m \033[97mProcessing error:\033[0m {e}")

        if self.debug:
            logger.debug(f"Processing thread stopped for client {client_id}")

    def _safe_send(self, websocket, message, loop):
        """Safely send message to websocket, handling connection errors"""
        try:
            future = asyncio.run_coroutine_threadsafe(websocket.send(message), loop)
            future.result(timeout=5.0)  # 5 second timeout
            return True
        except (
            websockets.exceptions.ConnectionClosed,
            websockets.exceptions.ConnectionClosedOK,
            websockets.exceptions.ConnectionClosedError,
            asyncio.TimeoutError,
            Exception,
        ) as e:
            if self.debug and not isinstance(
                e, (websockets.exceptions.ConnectionClosedOK, asyncio.TimeoutError)
            ):
                # Only log if it's not a "no close frame" error (common in tests)
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
                    break  # Connection closed, stop streaming

            # Send completion with inference time
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

            # Track performance per client
            with self.clients_lock:
                if client_id in self.client_frame_counts:
                    self.client_frame_counts[client_id] += 1
                    frame_count = self.client_frame_counts[client_id]
                    if frame_count % 10 == 0 and self.debug:
                        logger.debug(
                            f"Client {client_id}: Processed {frame_count} inputs, last inference: {inference_time:.2f}s"
                        )

        finally:
            # Always remove generator from tracking
            with self.clients_lock:
                if (
                    client_id in self.client_generators
                    and generator in self.client_generators[client_id]
                ):
                    self.client_generators[client_id].remove(generator)

    def _process_image(self, data, websocket, loop, client_id, stop_event):
        """Process image/video frame input"""
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
                    self.max_tokens_image
                    if self.max_tokens_image
                    else self.config["maxOutputTokens"]
                )
                temperature = self.config["temperature"]
                top_p = self.config["topP"]

            # Use model lock for thread-safe inference
            with self.model_lock:
                # Get additional generation parameters
                with self.config_lock:
                    top_k = self.config["topK"]
                    repetition_penalty = 1.0  # Default no penalty

                    # Convert presence/frequency penalties to repetition penalty
                    # MLX uses a single repetition_penalty parameter
                    if self.config["presencePenalty"] != 0 or self.config["frequencyPenalty"] != 0:
                        # Combine penalties - use the stronger one
                        # Typical range for repetition_penalty is 1.0-1.5
                        penalty_strength = max(
                            abs(self.config["presencePenalty"]),
                            abs(self.config["frequencyPenalty"]),
                        )
                        # Map penalty range [0, 2] to repetition_penalty range [1.0, 1.5]
                        repetition_penalty = 1.0 + (penalty_strength * 0.25)

                response_generator = generate(
                    self.model,
                    self.processor,
                    prompt=data["prompt"],
                    image=image,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    repetition_context_size=20,  # Look back 20 tokens for repetition
                    stream=True,
                )

                # Track generator for cleanup
                with self.clients_lock:
                    if client_id in self.client_generators:
                        self.client_generators[client_id].append(response_generator)

            # Use common streaming method
            self._stream_response(
                response_generator, websocket, loop, client_id, stop_event, data, "image"
            )

        except Exception as e:
            self._safe_send(
                websocket,
                json.dumps({"type": "error", "error": str(e), "timestamp": data["timestamp"]}),
                loop,
            )

    def _process_text(self, data, websocket, loop, client_id, stop_event):
        """Process text-only input"""
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
                max_tokens = self.config["maxOutputTokens"]
                temperature = self.config["temperature"]
                top_p = self.config["topP"]
                top_k = self.config["topK"]

                # Convert presence/frequency penalties to repetition penalty
                repetition_penalty = 1.0  # Default no penalty
                if self.config["presencePenalty"] != 0 or self.config["frequencyPenalty"] != 0:
                    # Combine penalties - use the stronger one
                    penalty_strength = max(
                        abs(self.config["presencePenalty"]), abs(self.config["frequencyPenalty"])
                    )
                    # Map penalty range [0, 2] to repetition_penalty range [1.0, 1.5]
                    repetition_penalty = 1.0 + (penalty_strength * 0.25)

            # Generate response with streaming
            # Check if this is a vision model or text-only model
            with self.model_lock:
                try:
                    # Try vision model API first
                    response_generator = generate(
                        self.model,
                        self.processor,
                        prompt=prompt,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        repetition_penalty=repetition_penalty,
                        repetition_context_size=20,
                        stream=True,
                    )
                except Exception:
                    # If vision API fails, try text-only API
                    # This allows compatibility with non-multimodal models
                    if text_generate is not None:
                        response_generator = text_generate(
                            self.model,
                            self.processor,
                            prompt=prompt,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            top_p=top_p,
                            top_k=top_k,
                            repetition_penalty=repetition_penalty,
                            repetition_context_size=20,
                            stream=True,
                        )
                    else:
                        raise Exception(
                            "Text-only model API not available and vision API failed"
                        ) from None

                # Track generator for cleanup
                with self.clients_lock:
                    if client_id in self.client_generators:
                        self.client_generators[client_id].append(response_generator)

            # Use common streaming method
            self._stream_response(
                response_generator, websocket, loop, client_id, stop_event, data, "text"
            )

        except Exception as e:
            self._safe_send(
                websocket,
                json.dumps({"type": "error", "error": str(e), "timestamp": data["timestamp"]}),
                loop,
            )

    async def _handle_config(self, config_update, websocket):
        """Handle configuration updates following formal schema

        Config fields (all optional):
        - candidateCount: Number of responses (MLX typically supports 1)
        - maxOutputTokens: Max tokens in response
        - temperature: Randomness control [0.0, 2.0]
        - topP: Cumulative probability for sampling
        - topK: Max tokens to consider when sampling
        - presencePenalty: Binary penalty for token reuse
        - frequencyPenalty: Scaled penalty based on usage count
        - responseModalities: ["TEXT", "IMAGE", "AUDIO"]
        """
        updated = {}

        # Validate and update each field if provided
        if "candidateCount" in config_update:
            try:
                value = int(config_update["candidateCount"])
                # MLX typically only supports 1 candidate
                if value != 1:
                    if self.debug:
                        logger.warning("MLX models typically support candidateCount=1 only")
                with self.config_lock:
                    self.config["candidateCount"] = value
                updated["candidateCount"] = value
            except (ValueError, TypeError):
                if self.debug:
                    logger.warning(
                        f"Invalid candidateCount value: {config_update['candidateCount']}"
                    )

        if "maxOutputTokens" in config_update:
            try:
                value = int(config_update["maxOutputTokens"])
                if value > 0:
                    with self.config_lock:
                        self.config["maxOutputTokens"] = value
                    updated["maxOutputTokens"] = value
                else:
                    if self.debug:
                        logger.warning(f"Invalid maxOutputTokens value: {value} (must be positive)")
            except (ValueError, TypeError):
                if self.debug:
                    logger.warning(
                        f"Invalid maxOutputTokens value: {config_update['maxOutputTokens']}"
                    )

        if "temperature" in config_update:
            try:
                value = float(config_update["temperature"])
                # Clamp temperature to valid range [0.0, 2.0]
                temp = max(0.0, min(2.0, value))
                with self.config_lock:
                    self.config["temperature"] = temp
                updated["temperature"] = temp
            except (ValueError, TypeError):
                if self.debug:
                    logger.warning(f"Invalid temperature value: {config_update['temperature']}")

        if "topP" in config_update:
            try:
                value = float(config_update["topP"])
                # Clamp topP to valid range [0.0, 1.0]
                top_p = max(0.0, min(1.0, value))
                with self.config_lock:
                    self.config["topP"] = top_p
                updated["topP"] = top_p
            except (ValueError, TypeError):
                if self.debug:
                    logger.warning(f"Invalid topP value: {config_update['topP']}")

        if "topK" in config_update:
            try:
                value = int(config_update["topK"])
                top_k = max(1, value)
                with self.config_lock:
                    self.config["topK"] = top_k
                updated["topK"] = top_k
            except (ValueError, TypeError):
                if self.debug:
                    logger.warning(f"Invalid topK value: {config_update['topK']}")

        if "presencePenalty" in config_update:
            try:
                value = float(config_update["presencePenalty"])
                with self.config_lock:
                    self.config["presencePenalty"] = value
                updated["presencePenalty"] = value
            except (ValueError, TypeError):
                if self.debug:
                    logger.warning(
                        f"Invalid presencePenalty value: {config_update['presencePenalty']}"
                    )

        if "frequencyPenalty" in config_update:
            try:
                value = float(config_update["frequencyPenalty"])
                with self.config_lock:
                    self.config["frequencyPenalty"] = value
                updated["frequencyPenalty"] = value
            except (ValueError, TypeError):
                if self.debug:
                    logger.warning(
                        f"Invalid frequencyPenalty value: {config_update['frequencyPenalty']}"
                    )

        if "responseModalities" in config_update:
            try:
                # Validate modalities
                if isinstance(config_update["responseModalities"], list):
                    valid_modalities = ["TEXT", "IMAGE", "AUDIO"]
                    modalities = [
                        m for m in config_update["responseModalities"] if m in valid_modalities
                    ]

                    # Note: Current implementation only supports TEXT
                    if any(m != "TEXT" for m in modalities) and self.debug:
                        logger.warning("Currently only TEXT modality is supported")

                    with self.config_lock:
                        self.config["responseModalities"] = modalities
                    updated["responseModalities"] = modalities
                else:
                    if self.debug:
                        logger.warning("Invalid responseModalities value: must be a list")
            except Exception as e:
                if self.debug:
                    logger.error(f"Error processing responseModalities: {e}")

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

        # Send confirmation with updated fields and full current config
        with self.config_lock:
            current_config = self.config.copy()

        await websocket.send(
            json.dumps(
                {
                    "type": "config_updated",
                    "updated_fields": updated,
                    "current_config": current_config,
                }
            )
        )

    async def shutdown(self):
        """Gracefully shutdown the server"""
        if not self.debug:
            if RICH_AVAILABLE:
                with Progress(
                    SpinnerColumn(style="pink"),
                    TextColumn("[dim_cream]Shutting down server...[/dim_cream]"),
                    console=console,
                    transient=False,
                ) as progress:
                    task = progress.add_task("shutdown", total=None)
                    # Brief delay to show the spinner
                    await asyncio.sleep(0.3)
            else:
                print("\n\n\033[93mShutting down server...\033[0m")
        else:
            logger.info("Shutting down server...")

        # Stop accepting new connections
        self.shutdown_event.set()

        # Close all active WebSocket connections
        if self.active_connections:
            if not self.debug:
                if RICH_AVAILABLE:
                    console.print(
                        f"[yellow]Closing {len(self.active_connections)} active connections...[/yellow]"
                    )
                else:
                    print(
                        f"\033[93mClosing {len(self.active_connections)} active connections...\033[0m"
                    )
            else:
                logger.info(f"Closing {len(self.active_connections)} active connections...")
            close_tasks = [ws.close() for ws in self.active_connections]
            await asyncio.gather(*close_tasks, return_exceptions=True)

        # Signal all client threads to stop
        with self.clients_lock:
            for stop_event in self.client_stop_events.values():
                stop_event.set()

        # Wait a bit for threads to finish
        await asyncio.sleep(0.5)

        # Close the server
        if self.server:
            self.server.close()
            await self.server.wait_closed()

        # Clean up MLX resources
        if self.model is not None:
            if not self.debug:
                if RICH_AVAILABLE:
                    console.print("[yellow]Cleaning up MLX model...[/yellow]")
                else:
                    print("\033[93mCleaning up MLX model...\033[0m")
            else:
                logger.info("Cleaning up MLX model...")
            # MLX doesn't have explicit cleanup, but we can clear references
            self.model = None
            self.processor = None
            # Force garbage collection to free MLX memory
            import gc

            gc.collect()
            if mx is not None:
                mx.metal.clear_cache()

        if not self.debug:
            if RICH_AVAILABLE:
                console.print("[green]✓[/green] [cream]Server shutdown complete[/cream]\n")
            else:
                print("\033[92m✓\033[0m \033[97mServer shutdown complete\033[0m\n")
        else:
            logger.info("Server shutdown complete")

    def _get_network_addresses(self):
        """Get all local network addresses"""
        addresses = []
        try:
            # Get hostname
            hostname = socket.gethostname()
            # Get all IPs associated with hostname
            host_ips = socket.gethostbyname_ex(hostname)[2]
            # Filter out loopback addresses
            for ip in host_ips:
                if not ip.startswith("127."):
                    addresses.append(ip)

            # Alternative method using socket connection
            if not addresses:
                with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                    # Connect to a public DNS server
                    s.connect(("8.8.8.8", 80))
                    addresses.append(s.getsockname()[0])
        except Exception:
            pass

        return addresses

    async def start_server(self):
        """Start the WebSocket server with graceful shutdown"""
        # Set up signal handlers
        loop = asyncio.get_event_loop()

        # Handle SIGINT (Ctrl+C) and SIGTERM
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, lambda s=sig: asyncio.create_task(self.shutdown()))

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
            # ANSI: purple=35 (magenta), white=97
            print("\n" + "\033[35m" + "=" * 60 + "\033[0m")
            print("\033[97mMLX WebSocket Streaming Server v0.1.0\033[0m")
            print("\033[35m" + "=" * 60 + "\033[0m" + "\n")

            print(f"\033[97mModel:\033[0m \033[95m{self.model_name}\033[0m")
            (
                print(
                    f"\033[97mMemory:\033[0m \033[95m{mx.get_active_memory() / 1024**3:.2f}\033[0m GB\n"
                )
                if mx
                else print("\033[97mMemory:\033[0m \033[35mNot available\033[0m\n")
            )

        try:
            self.server = await websockets.serve(self.handle_client, self.host, self.port)

            # Display connection URLs React-style
            if RICH_AVAILABLE:
                console.print("[cream]Server running at:[/cream]\n")
                console.print(
                    f"  [white]Local:[/white]            [cyan]ws://localhost:{self.port}[/cyan]"
                )

                # Get network addresses
                network_addresses = self._get_network_addresses()
                if network_addresses:
                    for ip in network_addresses:
                        console.print(
                            f"  [white]On Your Network:[/white]  [cyan]ws://{ip}:{self.port}[/cyan]"
                        )

                console.print(
                    "\n[cream]Note: To connect from other devices, ensure they are on the same network.[/cream]"
                )
                console.print(
                    "[white]Press[/white] [bold_white]Ctrl+C[/bold_white] [white]to stop the server.[/white]\n"
                )
            else:
                print("\033[97mServer running at:\033[0m\n")
                print(
                    f"  \033[97mLocal:\033[0m            \033[96mws://localhost:{self.port}\033[0m"
                )

                # Get network addresses
                network_addresses = self._get_network_addresses()
                if network_addresses:
                    for ip in network_addresses:
                        print(
                            f"  \033[97mOn Your Network:\033[0m  \033[96mws://{ip}:{self.port}\033[0m"
                        )

                print(
                    "\n\033[35mNote: To connect from other devices, ensure they are on the same network.\033[0m"
                )
                print(
                    "\033[97mPress\033[0m \033[95mCtrl+C\033[0m \033[97mto stop the server.\033[0m\n"
                )

            # Wait until shutdown is requested
            await self.shutdown_event.wait()

        except asyncio.CancelledError:
            if self.debug:
                logger.info("Server task cancelled")
        finally:
            # Ensure cleanup happens even on unexpected exit
            if not self.shutdown_event.is_set():
                await self.shutdown()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="MLX WebSocket Streaming Server - Real-time multimodal AI inference server for Apple Silicon Macs. "
        "Supports streaming video frames, images, and text chat with local MLX models.",
        epilog="Example: python mlx_streaming_server.py --model mlx-community/gemma-3-4b-it-4bit --port 8765",
    )
    parser.add_argument(
        "--model",
        default="mlx-community/gemma-3-4b-it-4bit",
        help="MLX model name from HuggingFace",
    )
    parser.add_argument("--port", type=int, default=8765, help="WebSocket server port")
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind to (use 'localhost' for local-only access)",
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
    if args.debug:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
    else:
        # Suppress most logs in production mode
        logging.basicConfig(
            level=logging.ERROR,
            format="%(message)s",
        )
        # Suppress transformers download messages
        logging.getLogger("transformers").setLevel(logging.ERROR)
        logging.getLogger("mlx").setLevel(logging.ERROR)
        logging.getLogger("websockets").setLevel(logging.ERROR)
        logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
        logging.getLogger("filelock").setLevel(logging.ERROR)

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
        model_name=args.model, port=args.port, host=args.host, debug=args.debug
    )
    asyncio.run(server.start_server())
