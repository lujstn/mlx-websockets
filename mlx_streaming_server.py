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
import threading
import time
from queue import Empty, Queue
from threading import Lock, RLock

import mlx.core as mx
import websockets
import websockets.exceptions
from mlx_vlm import generate, load
from PIL import Image

try:
    from mlx_lm import generate as text_generate
except ImportError:
    text_generate = None  # Will handle gracefully if not available

# Configure logging
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class MLXStreamingServer:
    def __init__(self, model_name="mlx-community/gemma-3-4b-it-4bit", port=8765):
        """
        Initialize the MLX WebSocket Streaming Server

        Args:
            model_name: MLX model from HuggingFace (Gemma 3 4bit model recommended)
            port: WebSocket server port (default 8765)
        """
        self.model_name = model_name
        self.port = port
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

        logger.info(f"Loading model: {model_name}")
        self._load_model()

    def _load_model(self):
        """Load the MLX model and processor"""
        try:
            self.model, self.processor = load(self.model_name)
            logger.info("Model loaded successfully!")
            logger.info(f"Memory usage: {mx.get_active_memory() / 1024**3:.2f} GB")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    async def handle_client(self, websocket, path):
        """Handle WebSocket client connections"""
        client_id = websocket.remote_address
        logger.info(f"Client connected: {client_id}")

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
            logger.info(f"Client disconnected: {client_id}")
        except Exception as e:
            logger.error(f"Error handling client {client_id}: {e}", exc_info=True)
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
                logger.error(f"Error processing input for client {client_id}: {e}")

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
            if not isinstance(e, (websockets.exceptions.ConnectionClosedOK, asyncio.TimeoutError)):
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
                    if frame_count % 10 == 0:
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
                    logger.warning("MLX models typically support candidateCount=1 only")
                with self.config_lock:
                    self.config["candidateCount"] = value
                updated["candidateCount"] = value
            except (ValueError, TypeError):
                logger.warning(f"Invalid candidateCount value: {config_update['candidateCount']}")

        if "maxOutputTokens" in config_update:
            try:
                value = int(config_update["maxOutputTokens"])
                if value > 0:
                    with self.config_lock:
                        self.config["maxOutputTokens"] = value
                    updated["maxOutputTokens"] = value
                else:
                    logger.warning(f"Invalid maxOutputTokens value: {value} (must be positive)")
            except (ValueError, TypeError):
                logger.warning(f"Invalid maxOutputTokens value: {config_update['maxOutputTokens']}")

        if "temperature" in config_update:
            try:
                value = float(config_update["temperature"])
                # Clamp temperature to valid range [0.0, 2.0]
                temp = max(0.0, min(2.0, value))
                with self.config_lock:
                    self.config["temperature"] = temp
                updated["temperature"] = temp
            except (ValueError, TypeError):
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
                logger.warning(f"Invalid topP value: {config_update['topP']}")

        if "topK" in config_update:
            try:
                value = int(config_update["topK"])
                top_k = max(1, value)
                with self.config_lock:
                    self.config["topK"] = top_k
                updated["topK"] = top_k
            except (ValueError, TypeError):
                logger.warning(f"Invalid topK value: {config_update['topK']}")

        if "presencePenalty" in config_update:
            try:
                value = float(config_update["presencePenalty"])
                with self.config_lock:
                    self.config["presencePenalty"] = value
                updated["presencePenalty"] = value
            except (ValueError, TypeError):
                logger.warning(f"Invalid presencePenalty value: {config_update['presencePenalty']}")

        if "frequencyPenalty" in config_update:
            try:
                value = float(config_update["frequencyPenalty"])
                with self.config_lock:
                    self.config["frequencyPenalty"] = value
                updated["frequencyPenalty"] = value
            except (ValueError, TypeError):
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
                    if any(m != "TEXT" for m in modalities):
                        logger.warning("Currently only TEXT modality is supported")

                    with self.config_lock:
                        self.config["responseModalities"] = modalities
                    updated["responseModalities"] = modalities
                else:
                    logger.warning("Invalid responseModalities value: must be a list")
            except Exception as e:
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
                    logger.warning(f"Invalid max_tokens_image value: {value} (must be positive)")
            except (ValueError, TypeError):
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
        logger.info("Shutting down server...")

        # Stop accepting new connections
        self.shutdown_event.set()

        # Close all active WebSocket connections
        if self.active_connections:
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
            logger.info("Cleaning up MLX model...")
            # MLX doesn't have explicit cleanup, but we can clear references
            self.model = None
            self.processor = None
            # Force garbage collection to free MLX memory
            import gc

            gc.collect()
            mx.metal.clear_cache()

        logger.info("Server shutdown complete")

    async def start_server(self):
        """Start the WebSocket server with graceful shutdown"""
        # Set up signal handlers
        loop = asyncio.get_event_loop()

        # Handle SIGINT (Ctrl+C) and SIGTERM
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, lambda s=sig: asyncio.create_task(self.shutdown()))

        logger.info(f"Starting WebSocket server on port {self.port}")
        logger.info("Press Ctrl+C to gracefully shutdown")

        try:
            self.server = await websockets.serve(self.handle_client, "localhost", self.port)
            logger.info(f"Server running at ws://localhost:{self.port}")

            # Wait until shutdown is requested
            await self.shutdown_event.wait()

        except asyncio.CancelledError:
            logger.info("Server task cancelled")
        finally:
            # Ensure cleanup happens even on unexpected exit
            if not self.shutdown_event.is_set():
                await self.shutdown()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MLX WebSocket Streaming Server")
    parser.add_argument(
        "--model",
        default="mlx-community/gemma-3-4b-it-4bit",
        help="MLX model name from HuggingFace",
    )
    parser.add_argument("--port", type=int, default=8765, help="WebSocket server port")

    args = parser.parse_args()

    # Start server
    server = MLXStreamingServer(model_name=args.model, port=args.port)
    asyncio.run(server.start_server())
