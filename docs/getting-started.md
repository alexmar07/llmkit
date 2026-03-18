# Getting Started

## Installation

```bash
pip install llmwire
```

Requires Python 3.12 or later.

## Configuration

### Direct configuration

Pass provider credentials and options directly when constructing `LLMConfig`:

```python
from llmwire import LLMConfig, ProviderConfig

config = LLMConfig(
    providers=[
        ProviderConfig(name="openai", api_key="sk-...", model="gpt-4o"),
    ],
    fallback=True,
    max_retries=3,
    timeout=30.0,
)
```

`ProviderConfig` fields:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | `str` | required | Provider name: `"openai"`, `"anthropic"`, or `"ollama"` |
| `api_key` | `str \| None` | `None` | API key (not required for Ollama) |
| `model` | `str` | `""` | Model identifier |
| `base_url` | `str \| None` | `None` | Override the default API endpoint |

`LLMConfig` fields:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `providers` | `list[ProviderConfig]` | `[]` | Ordered provider list |
| `fallback` | `bool` | `True` | Try next provider on failure |
| `max_retries` | `int` | `3` | Retry attempts per provider |
| `timeout` | `float` | `30.0` | HTTP request timeout in seconds |

### Environment variable configuration

`LLMConfig` is a `pydantic-settings` model with the prefix `LLMKIT_`. Nested fields
use `__` as a delimiter and providers are indexed starting from `0`:

```bash
export LLMKIT_PROVIDERS__0__NAME=openai
export LLMKIT_PROVIDERS__0__API_KEY=sk-...
export LLMKIT_PROVIDERS__0__MODEL=gpt-4o

export LLMKIT_PROVIDERS__1__NAME=anthropic
export LLMKIT_PROVIDERS__1__API_KEY=sk-ant-...
export LLMKIT_PROVIDERS__1__MODEL=claude-3-5-sonnet-20241022

export LLMKIT_FALLBACK=true
export LLMKIT_MAX_RETRIES=3
export LLMKIT_TIMEOUT=30.0
```

Then load config with no arguments:

```python
config = LLMConfig()
```

## Usage

### Basic chat

```python
import asyncio
from llmwire import LLMClient, LLMConfig, ProviderConfig

config = LLMConfig(
    providers=[ProviderConfig(name="openai", api_key="sk-...", model="gpt-4o")]
)

async def main():
    async with LLMClient(config) as client:
        response = await client.chat("Explain async/await in Python in one sentence.")
        print(response.content)

asyncio.run(main())
```

You can also pass a list of `Message` objects for multi-turn conversations:

```python
from llmwire import Message

messages = [
    Message(role="system", content="You are a concise assistant."),
    Message(role="user", content="What is Python?"),
    Message(role="assistant", content="A high-level, interpreted programming language."),
    Message(role="user", content="Who created it?"),
]

response = await client.chat(messages)
```

### Streaming

`client.stream()` returns an async generator of `StreamChunk` objects:

```python
async def main():
    async with LLMClient(config) as client:
        async for chunk in client.stream("Write a haiku about async programming."):
            print(chunk.content, end="", flush=True)
        print()

asyncio.run(main())
```

Each `StreamChunk` has the same fields as `ChatResponse` (`content`, `provider`, `model`, `usage`)
plus a `done: bool` flag that is `True` on the final chunk (Ollama only; other providers
signal end-of-stream by stopping iteration).

### Structured output

Pass any Pydantic `BaseModel` subclass as `response_model` to get a typed, validated
response object:

```python
import asyncio
from pydantic import BaseModel
from llmwire import LLMClient, LLMConfig, ProviderConfig

class MovieReview(BaseModel):
    title: str
    year: int
    rating: float
    summary: str

config = LLMConfig(
    providers=[ProviderConfig(name="openai", api_key="sk-...", model="gpt-4o")]
)

async def main():
    async with LLMClient(config) as client:
        review: MovieReview = await client.chat(
            "Review the movie Inception",
            response_model=MovieReview,
        )
        print(f"{review.title} ({review.year}) — {review.rating}/10")
        print(review.summary)

asyncio.run(main())
```

LLMWire prepends a system message with the Pydantic JSON schema and instructs the model
to respond with raw JSON. The response is then parsed with `model.model_validate_json()`.

### Using Ollama (local models)

Ollama requires no API key. Set `base_url` if your server is not at
`http://localhost:11434`:

```python
config = LLMConfig(
    providers=[
        ProviderConfig(name="ollama", model="llama3.2"),
    ]
)
```

### Fallback across providers

When `fallback=True` (the default), LLMWire tries each provider in order until one
succeeds. This is useful for resilience:

```python
config = LLMConfig(
    providers=[
        ProviderConfig(name="openai", api_key="sk-...", model="gpt-4o"),
        ProviderConfig(name="anthropic", api_key="sk-ant-...", model="claude-3-5-sonnet-20241022"),
        ProviderConfig(name="ollama", model="llama3.2"),
    ],
    fallback=True,
)
```

If all providers fail, `AllProvidersFailedError` is raised with the list of
individual `ProviderError` instances attached as `.errors`.

## Error Handling

```python
from llmwire import AllProvidersFailedError, ProviderError

try:
    response = await client.chat("Hello")
except AllProvidersFailedError as exc:
    for err in exc.errors:
        print(f"{err.provider}: {err}")
except ProviderError as exc:
    print(f"Provider {exc.provider} failed: {exc}")
```
