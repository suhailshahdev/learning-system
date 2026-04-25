"""LLM transport layer.

Defines the `LLMTransport` Protocol that every LLM provider implements,
plus the shared `TransportResponse` return type. Concrete transports
live alongside this module: `playwright_impl.py` for the claude.ai
browser automation transport, `deepseek_impl.py` for the DeepSeek
chat completions API transport.
"""

from app.transport.base import LLMTransport, TransportResponse

__all__ = [
    "LLMTransport",
    "TransportResponse",
]
