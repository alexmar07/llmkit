"""LLMClient — the main orchestrator for multi-provider LLM access."""
from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, TypeVar, cast, overload

from llmkit.exceptions import AllProvidersFailedError, ProviderError
from llmkit.models import ChatResponse, Message, StreamChunk
from llmkit.providers import AnthropicProvider, OllamaProvider, OpenAIProvider
from llmkit.retry import retry_with_backoff

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from pydantic import BaseModel

    from llmkit.config import LLMConfig, ProviderConfig

T = TypeVar("T", bound="BaseModel")


_schema_cache: dict[type[Any], str] = {}


def _cached_schema_json(model_cls: type[Any]) -> str:
    """Return the JSON schema string for a Pydantic model, cached per class."""
    if model_cls not in _schema_cache:
        _schema_cache[model_cls] = json.dumps(model_cls.model_json_schema(), indent=2)
    return _schema_cache[model_cls]

_PROVIDER_MAP: dict[str, type[Any]] = {
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "ollama": OllamaProvider,
}


def _create_provider(
    config: ProviderConfig, timeout: float
) -> OpenAIProvider | AnthropicProvider | OllamaProvider:
    """Instantiate a provider from its configuration.

    Args:
        config: Provider configuration including name, api_key, model, base_url.
        timeout: Request timeout in seconds forwarded to the provider.

    Returns:
        A concrete provider instance.

    Raises:
        ValueError: If the provider name is not recognised.
    """
    provider_cls = _PROVIDER_MAP.get(config.name)
    if provider_cls is None:
        known = ", ".join(_PROVIDER_MAP)
        raise ValueError(
            f"Unknown provider '{config.name}'. Known providers: {known}"
        )

    kwargs: dict[str, Any] = {"model": config.model, "timeout": timeout}

    if config.api_key is not None:
        kwargs["api_key"] = config.api_key

    if config.base_url is not None:
        kwargs["base_url"] = config.base_url

    instance = provider_cls(**kwargs)
    return cast("OpenAIProvider | AnthropicProvider | OllamaProvider", instance)


class LLMClient:
    """Orchestrates multi-provider LLM access with fallback and retry logic.

    Args:
        config: Full LLMKit configuration including provider list, fallback
            behaviour, retry count, and timeout.

    Raises:
        ValueError: If ``config.providers`` is empty.

    Example::

        config = LLMConfig(providers=[ProviderConfig(name="openai", api_key="...", model="gpt-4o")])
        client = LLMClient(config)
        response = await client.chat("Hello!")
        print(response.content)
    """

    def __init__(self, config: LLMConfig) -> None:
        if not config.providers:
            raise ValueError("At least one provider must be configured")
        self._config = config
        self._providers = [_create_provider(p, config.timeout) for p in config.providers]

    async def close(self) -> None:
        """Close all underlying provider HTTP clients."""
        for provider in self._providers:
            await provider._client.aclose()

    async def __aenter__(self) -> LLMClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    def _normalize_messages(self, prompt: str | list[Message]) -> list[Message]:
        """Convert a bare string prompt into a single-item user message list.

        Args:
            prompt: Either a plain string or an already-formed list of Messages.

        Returns:
            A list of Message objects ready to send to a provider.
        """
        if isinstance(prompt, str):
            return [Message(role="user", content=prompt)]
        return list(prompt)

    def _build_schema_system_message(self, model_cls: type[Any]) -> Message:
        """Build a system message instructing the LLM to respond with valid JSON.

        Args:
            model_cls: A Pydantic ``BaseModel`` subclass whose schema to embed.

        Returns:
            A system ``Message`` containing the JSON schema.
        """
        schema = _cached_schema_json(model_cls)
        content = (
            "You must respond with valid JSON that matches the following JSON schema. "
            "Do not include any explanation or markdown — only the raw JSON object.\n\n"
            f"Schema:\n{schema}"
        )
        return Message(role="system", content=content)

    @overload
    async def chat(
        self,
        prompt: str | list[Message],
        *,
        temperature: float = ...,
        max_tokens: int | None = ...,
        response_model: None = ...,
    ) -> ChatResponse: ...

    @overload
    async def chat(
        self,
        prompt: str | list[Message],
        *,
        temperature: float = ...,
        max_tokens: int | None = ...,
        response_model: type[T],
    ) -> T: ...

    async def chat(
        self,
        prompt: str | list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        response_model: type[T] | None = None,
    ) -> ChatResponse | T:
        """Send a chat completion request, with optional fallback across providers.

        Args:
            prompt: A string or list of ``Message`` objects. A bare string is
                wrapped in a single ``user`` message.
            temperature: Sampling temperature forwarded to the provider.
            max_tokens: Maximum tokens to generate; ``None`` uses provider default.
            response_model: Optional Pydantic model class. When supplied, a system
                message with the JSON schema is prepended and the response content
                is parsed into an instance of this model.

        Returns:
            A ``ChatResponse`` when ``response_model`` is ``None``, otherwise an
            instance of ``response_model``.

        Raises:
            AllProvidersFailedError: When all attempted providers raise
                ``ProviderError``.
        """
        messages = self._normalize_messages(prompt)

        if response_model is not None:
            schema_msg = self._build_schema_system_message(response_model)
            messages = [schema_msg, *messages]

        providers_to_try = self._providers if self._config.fallback else self._providers[:1]

        errors: list[ProviderError] = []
        for provider in providers_to_try:

            async def _call(
                _provider: OpenAIProvider | AnthropicProvider | OllamaProvider = provider,
            ) -> ChatResponse:
                return await _provider.chat(
                    messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

            try:
                response = await retry_with_backoff(
                    _call,
                    max_retries=self._config.max_retries,
                    retryable_exceptions=(ProviderError,),
                )
            except ProviderError as exc:
                errors.append(exc)
                continue

            if response_model is not None:
                return response_model.model_validate_json(response.content)
            return response

        raise AllProvidersFailedError(errors)

    async def stream(
        self,
        prompt: str | list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Stream a chat completion response, with optional fallback across providers.

        Yields chunks from the first provider that responds successfully.
        If a provider fails before streaming starts, the next provider is tried
        (when ``fallback=True``).

        Args:
            prompt: A string or list of ``Message`` objects.
            temperature: Sampling temperature forwarded to the provider.
            max_tokens: Maximum tokens to generate; ``None`` uses provider default.

        Yields:
            ``StreamChunk`` objects as tokens arrive.

        Raises:
            AllProvidersFailedError: When all attempted providers raise
                ``ProviderError``.
        """
        messages = self._normalize_messages(prompt)
        providers_to_try = self._providers if self._config.fallback else self._providers[:1]

        errors: list[ProviderError] = []
        for provider in providers_to_try:
            has_yielded = False
            try:
                async for chunk in provider.stream(
                    messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                ):
                    has_yielded = True
                    yield chunk
            except ProviderError as exc:
                if has_yielded:
                    # Already started yielding — cannot fall back; re-raise.
                    raise
                errors.append(exc)
                continue
            else:
                return

        raise AllProvidersFailedError(errors)
