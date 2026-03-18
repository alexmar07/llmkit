# LLMWire

[![CI](https://github.com/alexmar07/llmwire/actions/workflows/ci.yml/badge.svg)](https://github.com/alexmar07/llmwire/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/llmwire)](https://pypi.org/project/llmwire/)
[![Python](https://img.shields.io/pypi/pyversions/llmwire)](https://pypi.org/project/llmwire/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Lightweight multi-provider LLM client for Python. A single async interface to
OpenAI, Anthropic, and Ollama — with automatic fallback, exponential-backoff retry,
streaming, and structured Pydantic output. No provider SDK dependencies; all requests
go over plain `httpx`.

## Features

- **Unified API** — one `LLMClient` for all supported providers
- **Async-first** — built entirely on `asyncio` and `httpx`
- **Automatic fallback** — on provider failure, tries the next provider in the list
- **Exponential backoff** — configurable retry with full jitter
- **Streaming** — token-by-token via `client.stream()`, async generator interface
- **Structured output** — pass any Pydantic `BaseModel` as `response_model`
- **No provider SDKs** — runtime deps are only `httpx`, `pydantic`, `pydantic-settings`, `pyyaml`
- **Environment variable config** — all settings readable from `LLMKIT_*` env vars

## Quick Start

```bash
pip install llmwire
```

### Chat

```python
import asyncio
from llmwire import LLMClient, LLMConfig, ProviderConfig

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
        # Provider: openai | Model: gpt-4o | Tokens: 42

asyncio.run(main())
```

### Streaming

```python
async def main():
    async with LLMClient(config) as client:
        async for chunk in client.stream("Write a haiku about async programming."):
            print(chunk.content, end="", flush=True)
        print()
```

### Structured Output

```python
from pydantic import BaseModel

class Sentiment(BaseModel):
    label: str        # "positive", "negative", or "neutral"
    confidence: float

async def main():
    async with LLMClient(config) as client:
        result: Sentiment = await client.chat(
            "Classify: 'I love this library!'",
            response_model=Sentiment,
        )
        print(result.label, result.confidence)  # positive  0.97
```

## Configuration

### Direct

```python
from llmwire import LLMConfig, ProviderConfig

config = LLMConfig(
    providers=[
        ProviderConfig(name="openai", api_key="sk-...", model="gpt-4o"),
        ProviderConfig(name="ollama", model="llama3.2"),          # no key needed
    ],
    fallback=True,    # try next provider on failure (default: True)
    max_retries=3,    # per-provider retry attempts (default: 3)
    timeout=30.0,     # request timeout in seconds (default: 30.0)
)
```

### Environment Variables

```bash
export LLMKIT_PROVIDERS__0__NAME=openai
export LLMKIT_PROVIDERS__0__API_KEY=sk-...
export LLMKIT_PROVIDERS__0__MODEL=gpt-4o

export LLMKIT_PROVIDERS__1__NAME=anthropic
export LLMKIT_PROVIDERS__1__API_KEY=sk-ant-...
export LLMKIT_PROVIDERS__1__MODEL=claude-3-5-sonnet-20241022

export LLMKIT_FALLBACK=true
export LLMKIT_MAX_RETRIES=3
```

```python
config = LLMConfig()  # reads from environment
```

## Provider Support

| Provider | Chat | Streaming | Auth | Default endpoint |
|----------|------|-----------|------|-----------------|
| OpenAI | yes | yes | API key | `https://api.openai.com/v1` |
| Anthropic | yes | yes | API key | `https://api.anthropic.com/v1` |
| Ollama | yes | yes | none | `http://localhost:11434` |

The `base_url` field on `ProviderConfig` lets you point any provider at a compatible
endpoint (e.g. Azure OpenAI, local OpenAI-compatible servers).

## Further Reading

- [ARCHITECTURE.md](ARCHITECTURE.md) — design decisions, component overview, and provider protocol
- [CONTRIBUTING.md](CONTRIBUTING.md) — dev setup, code style, and how to add a new provider
- [Documentation](https://alexmar07.github.io/llmwire) — full API reference and guides

## License

MIT. See [LICENSE](LICENSE).
