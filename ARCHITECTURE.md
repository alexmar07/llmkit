# Architecture

## Design Philosophy

LLMKit is built around three principles:

1. **Minimal dependencies.** The runtime requires only `httpx`, `pydantic`,
   `pydantic-settings`, and `pyyaml`. There are no provider SDKs. Each provider
   adapter speaks the raw HTTP API directly, keeping the install small and the
   upgrade surface narrow.

2. **Async-first.** Every I/O path is async (`httpx.AsyncClient`, `async for` streaming).
   There is no sync wrapper; callers use `asyncio.run()` or their own event loop.

3. **Protocol-based extensibility.** Providers satisfy a structural `Protocol` rather
   than inheriting from a base class. This keeps provider implementations independent
   and makes it easy to add new ones without touching core logic.

---

## Component Overview

```
                    ┌─────────────────────────────────────┐
                    │             LLMClient                │
                    │  ┌──────────────────────────────┐   │
                    │  │  chat() / stream()            │   │
                    │  │  - normalize messages         │   │
                    │  │  - build schema system msg    │   │
                    │  │  - iterate providers_to_try   │   │
                    │  └──────────────────┬───────────┘   │
                    │                     │               │
                    │         retry_with_backoff()        │
                    │   (exponential backoff + jitter)    │
                    │                     │               │
                    │        ┌────────────┴──────────┐    │
                    │        │                       │    │
                    │   OpenAI         Anthropic    Ollama │
                    │  Provider        Provider   Provider │
                    │  (httpx)         (httpx)    (httpx)  │
                    └─────────────────────────────────────┘
```

**`LLMClient`** (`src/llmkit/client.py`) is the only public entry point. It holds an
ordered list of provider instances built from `LLMConfig` at construction time. `chat()`
and `stream()` iterate that list, delegating each attempt to `retry_with_backoff()`.

**`retry_with_backoff`** (`src/llmkit/retry.py`) is a standalone async function that
runs a callable up to `max_retries` times. Between attempts it sleeps for
`base_delay * 2^attempt + jitter` seconds. It only retries exceptions in the
`retryable_exceptions` tuple; other exceptions propagate immediately.

**Provider adapters** (`src/llmkit/providers/`) are plain classes. Each constructs a
single `httpx.AsyncClient` in `__init__` and uses it for all requests. `LLMClient`
calls `await provider._client.aclose()` in `close()` / `__aexit__`.

---

## Provider Protocol

`Provider` (`src/llmkit/provider.py`) is a `typing.Protocol`:

```python
class Provider(Protocol):
    @property
    def name(self) -> str: ...

    async def chat(
        self, messages: list[Message], *, model: str | None, temperature: float,
        max_tokens: int | None,
    ) -> ChatResponse: ...

    async def stream(
        self, messages: list[Message], *, model: str | None, temperature: float,
        max_tokens: int | None,
    ) -> AsyncIterator[StreamChunk]: ...
```

Any class that satisfies this interface structurally can be used as a provider.
`_PROVIDER_MAP` in `client.py` maps the string name from `ProviderConfig.name` to the
concrete class. Adding a new provider means:

1. Creating the adapter class in `src/llmkit/providers/`.
2. Registering it in `_PROVIDER_MAP`.
3. Exporting it from `src/llmkit/providers/__init__.py`.

---

## Retry and Fallback Strategy

`LLMConfig.fallback` controls whether `LLMClient` tries more than one provider:

- `fallback=False` — only `self._providers[0]` is tried.
- `fallback=True` (default) — all providers are tried in order until one succeeds.

Within each provider attempt, `retry_with_backoff()` retries `ProviderError` up to
`max_retries` times. The delay sequence for `base_delay=1.0` is roughly:
`1 s → 2 s → 4 s` (plus up to 10 % random jitter each time).

If a streaming response has already started yielding chunks, a mid-stream `ProviderError`
is re-raised immediately — falling back mid-stream would produce a corrupt response.

If all providers exhaust their retries, `AllProvidersFailedError` is raised with the
list of individual `ProviderError` instances accessible as `.errors`.

---

## Structured Output

`LLMClient.chat(..., response_model=MyModel)` works as follows:

1. The JSON schema of `MyModel` is serialised once and cached in `_schema_cache`
   (keyed by class identity).
2. A system `Message` is prepended instructing the model to return only raw JSON
   matching the schema.
3. The response content string is passed to `MyModel.model_validate_json()`, which
   raises `pydantic.ValidationError` on malformed output.

This approach is provider-agnostic and requires no function-calling support from the
underlying API.

---

## File Structure

```
llmkit/
├── src/llmkit/
│   ├── __init__.py          # public re-exports and __version__
│   ├── client.py            # LLMClient — main orchestrator
│   ├── config.py            # LLMConfig, ProviderConfig (pydantic-settings)
│   ├── exceptions.py        # LLMKitError, ProviderError, AllProvidersFailedError
│   ├── models.py            # Message, ChatResponse, StreamChunk, Usage
│   ├── provider.py          # Provider protocol
│   ├── retry.py             # retry_with_backoff()
│   └── providers/
│       ├── __init__.py
│       ├── anthropic.py     # AnthropicProvider
│       ├── ollama.py        # OllamaProvider
│       └── openai.py        # OpenAIProvider
├── tests/                   # pytest test suite (46 tests)
├── docs/                    # MkDocs source
├── .github/workflows/ci.yml # GitHub Actions CI/CD
├── pyproject.toml           # build config, dependencies, tool config
└── mkdocs.yml               # documentation site config
```
