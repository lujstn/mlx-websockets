"""Custom exceptions for mlx-websockets."""


class MLXWebSocketError(Exception):
    """Base exception for MLX WebSocket errors."""

    pass


class ModelLoadError(MLXWebSocketError):
    """Raised when model loading fails."""

    pass


class ClientConnectionError(MLXWebSocketError):
    """Raised when client connection issues occur."""

    pass


class MessageProcessingError(MLXWebSocketError):
    """Raised when message processing fails."""

    pass


class ImageProcessingError(MLXWebSocketError):
    """Raised when image processing fails."""

    pass


class TextGenerationError(MLXWebSocketError):
    """Raised when text generation fails."""

    pass


class ConfigurationError(MLXWebSocketError):
    """Raised when configuration is invalid."""

    pass


class ResourceError(MLXWebSocketError):
    """Raised when system resources are insufficient."""

    pass


class DaemonError(MLXWebSocketError):
    """Raised when daemon operations fail."""

    pass


class NetworkError(MLXWebSocketError):
    """Raised when network operations fail."""

    pass
