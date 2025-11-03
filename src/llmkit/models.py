"""Data models for LLMKit."""
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


class ChatResponse(BaseModel):
    """Response from an LLM chat completion."""

    content: str
    provider: str
    model: str
    usage: Usage | None = None


class StreamChunk(BaseModel):
    """A single chunk from a streaming response."""

    content: str
    provider: str
    model: str
    done: bool = False
    usage: Usage | None = None
