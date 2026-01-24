"""OpenAI provider implementation."""
from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import httpx

from llmkit.exceptions import ProviderError
from llmkit.models import ChatResponse, Message, StreamChunk, Usage

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


class OpenAIProvider:
    """LLM provider for OpenAI-compatible APIs.

    Uses httpx directly (no SDK dependency). Supports both chat completions
    and streaming via Server-Sent Events.

    Args:
        api_key: OpenAI API key.
        model: Model identifier (e.g. "gpt-4o").
        base_url: API base URL. Defaults to the official OpenAI endpoint.
        timeout: Request timeout in seconds. Defaults to 30.
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str = "https://api.openai.com/v1",
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
        return "openai"

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
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
        payload: dict[str, Any] = {
            "model": model or self._model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
            "stream": stream,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
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
            messages: Conversation history.
            model: Override the default model.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.

        Returns:
            ChatResponse with content, provider, model, and usage.

        Raises:
            ProviderError: On any non-200 HTTP response.
        """
        url = f"{self._base_url}/chat/completions"
        payload = self._build_payload(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        try:
            response = await self._client.post(url, headers=self._headers(), json=payload)
        except httpx.TransportError as exc:
            raise ProviderError(self.name, f"Connection failed: {exc}") from exc

        if response.status_code != 200:
            raise ProviderError(self.name, f"HTTP {response.status_code}: {response.text}")

        data = response.json()
        content: str = data["choices"][0]["message"]["content"]
        resolved_model: str = data.get("model", model or self._model)

        usage: Usage | None = None
        if raw_usage := data.get("usage"):
            usage = Usage(
                prompt_tokens=raw_usage.get("prompt_tokens", 0),
                completion_tokens=raw_usage.get("completion_tokens", 0),
                total_tokens=raw_usage.get("total_tokens", 0),
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

        Args:
            messages: Conversation history.
            model: Override the default model.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.

        Yields:
            StreamChunk for each token delta received.

        Raises:
            ProviderError: On any non-200 HTTP response.
        """
        url = f"{self._base_url}/chat/completions"
        payload = self._build_payload(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )

        try:
            async with self._client.stream(
                "POST", url, headers=self._headers(), json=payload
            ) as response:
                if response.status_code != 200:
                    body = await response.aread()
                    raise ProviderError(
                        self.name, f"HTTP {response.status_code}: {body.decode()}"
                    )

                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    raw = line[len("data: "):]
                    if raw == "[DONE]":
                        yield StreamChunk(
                            content="",
                            provider=self.name,
                            model=model or self._model,
                            done=True,
                        )
                        return
                    try:
                        event = json.loads(raw)
                    except json.JSONDecodeError:
                        continue

                    delta_content: str = event["choices"][0]["delta"].get("content", "")
                    event_model: str = event.get("model", model or self._model)

                    if delta_content:
                        yield StreamChunk(
                            content=delta_content,
                            provider=self.name,
                            model=event_model,
                        )
        except httpx.TransportError as exc:
            raise ProviderError(self.name, f"Connection failed: {exc}") from exc
