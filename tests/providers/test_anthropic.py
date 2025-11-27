"""Tests for Anthropic provider."""
import httpx
import pytest
import respx

from llmkit.exceptions import ProviderError
from llmkit.models import Message
from llmkit.providers.anthropic import AnthropicProvider


@pytest.fixture
def provider() -> AnthropicProvider:
    return AnthropicProvider(api_key="sk-ant-test", model="claude-3-5-sonnet-20241022")


class TestAnthropicChat:
    @respx.mock
    async def test_chat_success(self, provider: AnthropicProvider) -> None:
        respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=httpx.Response(
                200,
                json={
                    "content": [{"type": "text", "text": "Hello!"}],
                    "model": "claude-3-5-sonnet-20241022",
                    "usage": {"input_tokens": 10, "output_tokens": 5},
                },
            )
        )
        response = await provider.chat([Message(role="user", content="Hi")])
        assert response.content == "Hello!"
        assert response.provider == "anthropic"
        assert response.model == "claude-3-5-sonnet-20241022"
        assert response.usage is not None
        assert response.usage.prompt_tokens == 10
        assert response.usage.completion_tokens == 5
        assert response.usage.total_tokens == 15

    @respx.mock
    async def test_chat_system_message_extraction(self, provider: AnthropicProvider) -> None:
        """System messages must go into the top-level 'system' field, not in messages array."""
        route = respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=httpx.Response(
                200,
                json={
                    "content": [{"type": "text", "text": "Sure!"}],
                    "model": "claude-3-5-sonnet-20241022",
                    "usage": {"input_tokens": 20, "output_tokens": 8},
                },
            )
        )
        messages = [
            Message(role="system", content="You are a helpful assistant."),
            Message(role="user", content="Help me."),
        ]
        await provider.chat(messages)

        sent = route.calls.last.request
        import json

        body = json.loads(sent.content)
        assert body["system"] == "You are a helpful assistant."
        assert all(m["role"] != "system" for m in body["messages"])

    @respx.mock
    async def test_chat_api_error(self, provider: AnthropicProvider) -> None:
        respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=httpx.Response(401, json={"error": {"message": "Unauthorized"}})
        )
        with pytest.raises(ProviderError, match="anthropic"):
            await provider.chat([Message(role="user", content="Hi")])

    def test_provider_name(self, provider: AnthropicProvider) -> None:
        assert provider.name == "anthropic"


class TestAnthropicStream:
    @respx.mock
    async def test_stream_success(self, provider: AnthropicProvider) -> None:
        sse_data = (
            'event: content_block_delta\ndata: {"type":"content_block_delta","delta":{"type":"text_delta","text":"Hel"}}\n\n'
            'event: content_block_delta\ndata: {"type":"content_block_delta","delta":{"type":"text_delta","text":"lo"}}\n\n'
            'event: content_block_delta\ndata: {"type":"content_block_delta","delta":{"type":"text_delta","text":"!"}}\n\n'
            'event: message_stop\ndata: {"type":"message_stop"}\n\n'
        )
        respx.post("https://api.anthropic.com/v1/messages").mock(
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
