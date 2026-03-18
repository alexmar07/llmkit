"""Tests for OpenAI provider."""
import httpx
import pytest
import respx

from llmwire.exceptions import ProviderError
from llmwire.models import Message
from llmwire.providers.openai import OpenAIProvider


@pytest.fixture
def provider() -> OpenAIProvider:
    return OpenAIProvider(api_key="sk-test", model="gpt-4o")


class TestOpenAIChat:
    @respx.mock
    async def test_chat_success(self, provider: OpenAIProvider) -> None:
        respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": "Hello!"}}],
                    "model": "gpt-4o",
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                },
            )
        )
        response = await provider.chat([Message(role="user", content="Hi")])
        assert response.content == "Hello!"
        assert response.provider == "openai"
        assert response.model == "gpt-4o"
        assert response.usage is not None
        assert response.usage.total_tokens == 15

    @respx.mock
    async def test_chat_api_error(self, provider: OpenAIProvider) -> None:
        respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=httpx.Response(429, json={"error": {"message": "Rate limited"}})
        )
        with pytest.raises(ProviderError, match="openai"):
            await provider.chat([Message(role="user", content="Hi")])

    def test_provider_name(self, provider: OpenAIProvider) -> None:
        assert provider.name == "openai"


class TestOpenAIStream:
    @respx.mock
    async def test_stream_success(self, provider: OpenAIProvider) -> None:
        sse_data = (
            'data: {"choices":[{"delta":{"content":"Hel"}}],"model":"gpt-4o"}\n\n'
            'data: {"choices":[{"delta":{"content":"lo"}}],"model":"gpt-4o"}\n\n'
            'data: {"choices":[{"delta":{"content":"!"}}],"model":"gpt-4o"}\n\n'
            "data: [DONE]\n\n"
        )
        respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                content=sse_data.encode(),
                headers={"content-type": "text/event-stream"},
            )
        )
        chunks: list[str] = []
        async for chunk in provider.stream([Message(role="user", content="Hi")]):
            chunks.append(chunk.content)
        assert "".join(chunks) == "Hello!"
