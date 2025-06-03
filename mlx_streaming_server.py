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
import threading
import time
from queue import Empty, Queue
from threading import Lock, RLock

import mlx.core as mx
import websockets
from mlx_vlm import generate, load
from PIL import Image

try:
    from mlx_lm import generate as text_generate
except ImportError:
    text_generate = None  # Will handle gracefully if not available


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

        print(f"Loading model: {model_name}")
        self._load_model()

    def _load_model(self):
        """Load the MLX model and processor"""
        try:
            self.model, self.processor = load(self.model_name)
            print("Model loaded successfully!")
            print(f"Memory usage: {mx.metal.get_active_memory() / 1024**3:.2f} GB")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    async def handle_client(self, websocket, path):
        """Handle WebSocket client connections"""
        client_id = websocket.remote_address
        print(f"Client connected: {client_id}")

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
            daemon=True,
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
            print(f"Client disconnected: {client_id}")
        except Exception as e:
            print(f"Error handling client {client_id}: {e}")
        finally:
            # Signal thread to stop and clean up generators
            with self.clients_lock:
                if client_id in self.client_stop_events:
                    self.client_stop_events[client_id].set()

                # Cancel any active generators
                if client_id in self.client_generators:
                    # Clear the list to signal no more processing needed
                    self.client_generators[client_id].clear()

            # Wait a bit for thread to finish processing
            await asyncio.sleep(0.5)

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
                print(f"Error processing input for client {client_id}: {e}")

        print(f"Processing thread stopped for client {client_id}")

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
                print(f"Error sending message: {e}")
            return False

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

            # Track inference time
            inference_start = time.time()

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
                            abs(self.config["frequencyPenalty"])
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

            full_response = ""
            max_timeout = 60  # 60 seconds max for inference

            for token in response_generator:
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

            # Remove generator from tracking
            with self.clients_lock:
                if (
                    client_id in self.client_generators
                    and response_generator in self.client_generators[client_id]
                ):
                    self.client_generators[client_id].remove(response_generator)

            # Send completion with inference time
            inference_time = time.time() - inference_start
            self._safe_send(
                websocket,
                json.dumps(
                    {
                        "type": "response_complete",
                        "full_text": full_response,
                        "timestamp": data["timestamp"],
                        "input_type": "image",
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
                        print(
                            f"Client {client_id}: Processed {frame_count} inputs, last inference: {inference_time:.2f}s"
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

            # Track inference time
            inference_start = time.time()

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
                        abs(self.config["presencePenalty"]),
                        abs(self.config["frequencyPenalty"])
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
                        raise Exception("Text-only model API not available and vision API failed") from None

                # Track generator for cleanup
                with self.clients_lock:
                    if client_id in self.client_generators:
                        self.client_generators[client_id].append(response_generator)

            full_response = ""
            max_timeout = 60  # 60 seconds max for inference

            for token in response_generator:
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

            # Remove generator from tracking
            with self.clients_lock:
                if (
                    client_id in self.client_generators
                    and response_generator in self.client_generators[client_id]
                ):
                    self.client_generators[client_id].remove(response_generator)

            # Send completion with inference time
            inference_time = time.time() - inference_start
            self._safe_send(
                websocket,
                json.dumps(
                    {
                        "type": "response_complete",
                        "full_text": full_response,
                        "timestamp": data["timestamp"],
                        "input_type": "text",
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
                        print(
                            f"Client {client_id}: Processed {frame_count} inputs, last inference: {inference_time:.2f}s"
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
                    print("Warning: MLX models typically support candidateCount=1 only")
                with self.config_lock:
                    self.config["candidateCount"] = value
                updated["candidateCount"] = value
            except (ValueError, TypeError):
                print(f"Invalid candidateCount value: {config_update['candidateCount']}")

        if "maxOutputTokens" in config_update:
            try:
                value = int(config_update["maxOutputTokens"])
                if value > 0:
                    with self.config_lock:
                        self.config["maxOutputTokens"] = value
                    updated["maxOutputTokens"] = value
                else:
                    print(f"Invalid maxOutputTokens value: {value} (must be positive)")
            except (ValueError, TypeError):
                print(f"Invalid maxOutputTokens value: {config_update['maxOutputTokens']}")

        if "temperature" in config_update:
            try:
                value = float(config_update["temperature"])
                # Clamp temperature to valid range [0.0, 2.0]
                temp = max(0.0, min(2.0, value))
                with self.config_lock:
                    self.config["temperature"] = temp
                updated["temperature"] = temp
            except (ValueError, TypeError):
                print(f"Invalid temperature value: {config_update['temperature']}")

        if "topP" in config_update:
            try:
                value = float(config_update["topP"])
                # Clamp topP to valid range [0.0, 1.0]
                top_p = max(0.0, min(1.0, value))
                with self.config_lock:
                    self.config["topP"] = top_p
                updated["topP"] = top_p
            except (ValueError, TypeError):
                print(f"Invalid topP value: {config_update['topP']}")

        if "topK" in config_update:
            try:
                value = int(config_update["topK"])
                top_k = max(1, value)
                with self.config_lock:
                    self.config["topK"] = top_k
                updated["topK"] = top_k
            except (ValueError, TypeError):
                print(f"Invalid topK value: {config_update['topK']}")

        if "presencePenalty" in config_update:
            try:
                value = float(config_update["presencePenalty"])
                with self.config_lock:
                    self.config["presencePenalty"] = value
                updated["presencePenalty"] = value
            except (ValueError, TypeError):
                print(f"Invalid presencePenalty value: {config_update['presencePenalty']}")

        if "frequencyPenalty" in config_update:
            try:
                value = float(config_update["frequencyPenalty"])
                with self.config_lock:
                    self.config["frequencyPenalty"] = value
                updated["frequencyPenalty"] = value
            except (ValueError, TypeError):
                print(f"Invalid frequencyPenalty value: {config_update['frequencyPenalty']}")

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
                        print("Warning: Currently only TEXT modality is supported")

                    with self.config_lock:
                        self.config["responseModalities"] = modalities
                    updated["responseModalities"] = modalities
                else:
                    print("Invalid responseModalities value: must be a list")
            except Exception as e:
                print(f"Error processing responseModalities: {e}")

        # Special handling for convenience max_tokens_image
        if "max_tokens_image" in config_update:
            try:
                value = int(config_update["max_tokens_image"])
                if value > 0:
                    with self.config_lock:
                        self.max_tokens_image = value
                    updated["max_tokens_image"] = value
                else:
                    print(f"Invalid max_tokens_image value: {value} (must be positive)")
            except (ValueError, TypeError):
                print(f"Invalid max_tokens_image value: {config_update['max_tokens_image']}")

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

    async def start_server(self):
        """Start the WebSocket server"""
        print(f"Starting WebSocket server on port {self.port}")
        async with websockets.serve(self.handle_client, "localhost", self.port):
            print(f"Server running at ws://localhost:{self.port}")
            await asyncio.Future()  # Run forever


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
