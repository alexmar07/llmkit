# LLMKit

**Lightweight multi-provider LLM client for Python.**

LLMKit provides a single async interface to OpenAI, Anthropic, and Ollama. It handles
provider fallback, exponential-backoff retry, streaming, and structured output — with
no provider SDK dependencies. Everything goes over plain `httpx`.

## Quick Example

```python
import asyncio
from llmkit import LLMClient, LLMConfig, ProviderConfig

config = LLMConfig(
    providers=[
        ProviderConfig(name="openai", api_key="sk-...", model="gpt-4o"),
        ProviderConfig(name="anthropic", api_key="sk-ant-...", model="claude-3-5-sonnet-20241022"),
    ]
)

async def main():
    async with LLMClient(config) as client:
        response = await client.chat("What is the capital of France?")
        print(response.content)
        print(f"Provider: {response.provider}, tokens: {response.usage.total_tokens}")

asyncio.run(main())
```

## Key Features

- **Unified API** — one interface for all supported providers
- **Async-first** — built entirely on `asyncio` and `httpx`
- **Automatic fallback** — on failure, tries the next provider in the list
- **Exponential backoff** — configurable retry with jitter
- **Streaming** — token-by-token streaming via `client.stream()`
- **Structured output** — pass a Pydantic model as `response_model` to get typed output
- **No provider SDKs** — only `httpx`, `pydantic`, and `pydantic-settings`
- **Environment variable config** — all settings readable from `LLMKIT_*` env vars

## Navigation

- [Getting Started](getting-started.md) — installation, configuration, and usage examples
- [API Reference](api-reference.md) — full class and method documentation
