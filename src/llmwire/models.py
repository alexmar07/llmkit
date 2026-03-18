"""Data models for LLMWire."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

Role = Literal["system", "user", "assistant"]


class Message(BaseModel):
    """A chat message."""

    role: Role
    content: str


class Usage(BaseModel):
    """Token usage statistics."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class _LLMResponseBase(BaseModel):
    """Shared fields for all LLM response types."""

    content: str
    provider: str
    model: str
    usage: Usage | None = None


class ChatResponse(_LLMResponseBase):
    """Response from an LLM chat completion."""


class StreamChunk(_LLMResponseBase):
    """A single chunk from a streaming response."""

    done: bool = False
