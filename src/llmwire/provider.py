"""Provider protocol — the interface all LLM providers implement."""
from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from llmwire.models import ChatResponse, Message, StreamChunk


class Provider(Protocol):
    """Interface for LLM provider adapters."""

    @property
    def name(self) -> str: ...

    async def close(self) -> None: ...

    async def chat(
        self,
        messages: list[Message],
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> ChatResponse: ...

    async def stream(
        self,
        messages: list[Message],
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> AsyncIterator[StreamChunk]: ...
