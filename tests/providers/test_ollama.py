"""Tests for Ollama provider."""
import httpx
import pytest
import respx

from llmkit.exceptions import ProviderError
from llmkit.models import Message
from llmkit.providers.ollama import OllamaProvider


@pytest.fixture
def provider() -> OllamaProvider:
    return OllamaProvider(model="llama3.2")


class TestOllamaChat:
    @respx.mock
    async def test_chat_success(self, provider: OllamaProvider) -> None:
        respx.post("http://localhost:11434/api/chat").mock(
            return_value=httpx.Response(
                200,
                json={
                    "message": {"role": "assistant", "content": "Hello!"},
                    "model": "llama3.2",
                    "prompt_eval_count": 10,
                    "eval_count": 5,
                    "done": True,
                },
            )
        )
        response = await provider.chat([Message(role="user", content="Hi")])
        assert response.content == "Hello!"
        assert response.provider == "ollama"
        assert response.model == "llama3.2"
        assert response.usage is not None
        assert response.usage.prompt_tokens == 10
        assert response.usage.completion_tokens == 5
        assert response.usage.total_tokens == 15

    @respx.mock
    async def test_chat_connection_error(self, provider: OllamaProvider) -> None:
        respx.post("http://localhost:11434/api/chat").mock(
            side_effect=httpx.ConnectError("Connection refused")
        )
        with pytest.raises(ProviderError, match="ollama"):
            await provider.chat([Message(role="user", content="Hi")])

    @respx.mock
    async def test_chat_api_error(self, provider: OllamaProvider) -> None:
        respx.post("http://localhost:11434/api/chat").mock(
            return_value=httpx.Response(500, json={"error": "model not found"})
        )
        with pytest.raises(ProviderError, match="ollama"):
            await provider.chat([Message(role="user", content="Hi")])

    def test_provider_name(self, provider: OllamaProvider) -> None:
        assert provider.name == "ollama"

    def test_default_base_url(self, provider: OllamaProvider) -> None:
        assert provider.base_url == "http://localhost:11434"


class TestOllamaStream:
    @respx.mock
    async def test_stream_success(self, provider: OllamaProvider) -> None:
        import json

        ndjson = (
            json.dumps({"message": {"content": "Hel"}, "model": "llama3.2", "done": False})
            + "\n"
            + json.dumps({"message": {"content": "lo"}, "model": "llama3.2", "done": False})
            + "\n"
            + json.dumps({"message": {"content": "!"}, "model": "llama3.2", "done": False})
            + "\n"
            + json.dumps({"message": {"content": ""}, "model": "llama3.2", "done": True})
            + "\n"
        )
        respx.post("http://localhost:11434/api/chat").mock(
            return_value=httpx.Response(
                200,
                content=ndjson.encode(),
                headers={"content-type": "application/x-ndjson"},
            )
        )
        chunks: list[str] = []
        async for chunk in provider.stream([Message(role="user", content="Hi")]):
            chunks.append(chunk.content)
        assert "".join(chunks) == "Hello!"
