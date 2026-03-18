"""Ollama provider implementation for local LLM inference."""
from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import httpx

from llmwire.exceptions import ProviderError
from llmwire.models import ChatResponse, Message, StreamChunk, Usage

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

_DEFAULT_BASE_URL = "http://localhost:11434"
_DEFAULT_TIMEOUT = 120.0  # local models are slower


class OllamaProvider:
    """LLM provider for Ollama local inference server.

    Connects to a locally running Ollama instance. No API key is needed.
    Uses NDJSON for streaming (one JSON object per line).

    Args:
        model: Model identifier (e.g. "llama3.2").
        base_url: Ollama server URL. Defaults to ``http://localhost:11434``.
        timeout: Request timeout in seconds. Defaults to 120.
    """

    def __init__(
        self,
        model: str,
        base_url: str = _DEFAULT_BASE_URL,
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> None:
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._client = httpx.AsyncClient(timeout=self._timeout)

    @property
    def name(self) -> str:
        """Provider identifier."""
        return "ollama"

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()

    @property
    def base_url(self) -> str:
        """Ollama server base URL."""
        return self._base_url

    def _headers(self) -> dict[str, str]:
        return {}

    def _build_payload(
        self,
        messages: list[Message],
        *,
        model: str | None,
        temperature: float,
        max_tokens: int | None,
        stream: bool = False,
    ) -> dict[str, Any]:
        options: dict[str, Any] = {"temperature": temperature}
        if max_tokens is not None:
            options["num_predict"] = max_tokens

        return {
            "model": model or self._model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": stream,
            "options": options,
        }

    async def chat(
        self,
        messages: list[Message],
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> ChatResponse:
        """Send a chat completion request to the Ollama server.

        Args:
            messages: Conversation history.
            model: Override the default model.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate (``num_predict``).

        Returns:
            ChatResponse with content, provider, model, and usage.

        Raises:
            ProviderError: On connection errors or non-200 HTTP responses.
        """
        url = f"{self._base_url}/api/chat"
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
        content: str = data["message"]["content"]
        resolved_model: str = data.get("model", model or self._model)

        prompt_tokens: int = data.get("prompt_eval_count", 0)
        completion_tokens: int = data.get("eval_count", 0)
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
        """Stream a chat completion response via NDJSON.

        Ollama streams one JSON object per line. The ``done`` field signals
        the end of the stream.

        Args:
            messages: Conversation history.
            model: Override the default model.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate (``num_predict``).

        Yields:
            StreamChunk for each non-empty content token.

        Raises:
            ProviderError: On connection errors or non-200 HTTP responses.
        """
        url = f"{self._base_url}/api/chat"
        payload = self._build_payload(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        resolved_model = model or self._model

        try:
            async with self._client.stream(
                "POST", url, headers=self._headers(), json=payload
            ) as response:
                if response.status_code != 200:
                    body = await response.aread()
                    raise ProviderError(self.name, f"HTTP {response.status_code}: {body.decode()}")

                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    chunk_content: str = event.get("message", {}).get("content", "")
                    event_model: str = event.get("model", resolved_model)
                    is_done: bool = event.get("done", False)

                    if chunk_content:
                        yield StreamChunk(
                            content=chunk_content,
                            provider=self.name,
                            model=event_model,
                            done=is_done,
                        )

                    if is_done:
                        break
        except httpx.TransportError as exc:
            raise ProviderError(self.name, f"Connection failed: {exc}") from exc
