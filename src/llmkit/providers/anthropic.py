"""Anthropic provider implementation."""
from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import httpx

from llmkit.exceptions import ProviderError
from llmkit.models import ChatResponse, Message, StreamChunk, Usage

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

_ANTHROPIC_VERSION = "2023-06-01"


class AnthropicProvider:
    """LLM provider for Anthropic's Messages API.

    Uses httpx directly. Handles system message extraction into the
    top-level ``system`` field as required by the Anthropic API.
    Supports both chat completions and streaming via Server-Sent Events.

    Args:
        api_key: Anthropic API key.
        model: Model identifier (e.g. "claude-3-5-sonnet-20241022").
        base_url: API base URL. Defaults to the official Anthropic endpoint.
        timeout: Request timeout in seconds. Defaults to 30.
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str = "https://api.anthropic.com/v1",
        timeout: float = 30.0,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._client = httpx.AsyncClient(timeout=self._timeout)

    @property
    def name(self) -> str:
        """Provider identifier."""
        return "anthropic"

    def _headers(self) -> dict[str, str]:
        return {
            "x-api-key": self._api_key,
            "anthropic-version": _ANTHROPIC_VERSION,
        }

    def _build_payload(
        self,
        messages: list[Message],
        *,
        model: str | None,
        temperature: float,
        max_tokens: int | None,
        stream: bool = False,
    ) -> dict[str, Any]:
        """Build the request payload, extracting system messages."""
        system_parts: list[str] = []
        user_messages: list[dict[str, str]] = []

        for msg in messages:
            if msg.role == "system":
                system_parts.append(msg.content)
            else:
                user_messages.append({"role": msg.role, "content": msg.content})

        payload: dict[str, Any] = {
            "model": model or self._model,
            "messages": user_messages,
            "temperature": temperature,
            "max_tokens": max_tokens if max_tokens is not None else 4096,
            "stream": stream,
        }
        if system_parts:
            payload["system"] = "\n".join(system_parts)

        return payload

    async def chat(
        self,
        messages: list[Message],
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> ChatResponse:
        """Send a chat completion request.

        Args:
            messages: Conversation history. System messages are extracted
                and placed in the top-level ``system`` field.
            model: Override the default model.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate. Defaults to 4096.

        Returns:
            ChatResponse with content, provider, model, and usage.

        Raises:
            ProviderError: On any non-200 HTTP response.
        """
        url = f"{self._base_url}/messages"
        payload = self._build_payload(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        try:
            response = await self._client.post(url, headers=self._headers(), json=payload)
        except httpx.ConnectError as exc:
            raise ProviderError(self.name, f"Connection failed: {exc}") from exc

        if response.status_code != 200:
            raise ProviderError(self.name, f"HTTP {response.status_code}: {response.text}")

        data = response.json()
        content: str = data["content"][0]["text"]
        resolved_model: str = data.get("model", model or self._model)

        usage: Usage | None = None
        if raw_usage := data.get("usage"):
            prompt_tokens: int = raw_usage.get("input_tokens", 0)
            completion_tokens: int = raw_usage.get("output_tokens", 0)
            usage = Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            )

        return ChatResponse(
            content=content,
            provider=self.name,
            model=resolved_model,
            usage=usage,
        )

    async def stream(
        self,
        messages: list[Message],
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Stream a chat completion response via SSE.

        Anthropic uses typed events. Only ``content_block_delta`` events
        with ``text_delta`` deltas produce text. ``message_stop`` signals
        the end of the stream.

        Args:
            messages: Conversation history.
            model: Override the default model.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate. Defaults to 4096.

        Yields:
            StreamChunk for each text delta received.

        Raises:
            ProviderError: On any non-200 HTTP response.
        """
        url = f"{self._base_url}/messages"
        payload = self._build_payload(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        resolved_model = model or self._model

        async with self._client.stream(
            "POST", url, headers=self._headers(), json=payload
        ) as response:
            if response.status_code != 200:
                body = await response.aread()
                raise ProviderError(self.name, f"HTTP {response.status_code}: {body.decode()}")

            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                raw = line[len("data: "):]
                try:
                    event = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                event_type: str = event.get("type", "")

                if event_type == "message_stop":
                    break

                if event_type == "content_block_delta":
                    delta = event.get("delta", {})
                    if delta.get("type") == "text_delta":
                        text: str = delta.get("text", "")
                        if text:
                            yield StreamChunk(
                                content=text,
                                provider=self.name,
                                model=resolved_model,
                            )
