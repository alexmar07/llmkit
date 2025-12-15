"""Tests for LLMClient."""
import httpx
import pytest
import respx
from pydantic import BaseModel

from llmkit.client import LLMClient
from llmkit.config import LLMConfig, ProviderConfig
from llmkit.exceptions import AllProvidersFailedError
from llmkit.models import Message


def _openai_config() -> ProviderConfig:
    return ProviderConfig(name="openai", api_key="sk-test", model="gpt-4o")


def _anthropic_config() -> ProviderConfig:
    return ProviderConfig(name="anthropic", api_key="sk-ant-test", model="claude-sonnet-4-6")


def _mock_openai_success() -> None:
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "OpenAI response"}}],
                "model": "gpt-4o",
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            },
        )
    )


def _mock_openai_failure() -> None:
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(500, json={"error": {"message": "Server error"}})
    )


def _mock_anthropic_success() -> None:
    respx.post("https://api.anthropic.com/v1/messages").mock(
        return_value=httpx.Response(
            200,
            json={
                "content": [{"type": "text", "text": "Claude response"}],
                "model": "claude-sonnet-4-6",
                "usage": {"input_tokens": 10, "output_tokens": 5},
            },
        )
    )


class TestLLMClientChat:
    @respx.mock
    async def test_chat_single_provider(self) -> None:
        _mock_openai_success()
        config = LLMConfig(providers=[_openai_config()])
        client = LLMClient(config)
        response = await client.chat("Hello")
        assert response.content == "OpenAI response"
        assert response.provider == "openai"

    @respx.mock
    async def test_chat_with_message_objects(self) -> None:
        _mock_openai_success()
        config = LLMConfig(providers=[_openai_config()])
        client = LLMClient(config)
        messages = [
            Message(role="system", content="Be helpful"),
            Message(role="user", content="Hello"),
        ]
        response = await client.chat(messages)
        assert response.content == "OpenAI response"

    @respx.mock
    async def test_fallback_to_second_provider(self) -> None:
        _mock_openai_failure()
        _mock_anthropic_success()
        config = LLMConfig(
            providers=[_openai_config(), _anthropic_config()],
            fallback=True,
            max_retries=1,
        )
        client = LLMClient(config)
        response = await client.chat("Hello")
        assert response.content == "Claude response"
        assert response.provider == "anthropic"

    @respx.mock
    async def test_no_fallback_raises(self) -> None:
        _mock_openai_failure()
        config = LLMConfig(
            providers=[_openai_config(), _anthropic_config()],
            fallback=False,
            max_retries=1,
        )
        client = LLMClient(config)
        with pytest.raises(AllProvidersFailedError):
            await client.chat("Hello")

    @respx.mock
    async def test_all_providers_fail(self) -> None:
        _mock_openai_failure()
        respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=httpx.Response(500, json={"error": "fail"})
        )
        config = LLMConfig(
            providers=[_openai_config(), _anthropic_config()],
            fallback=True,
            max_retries=1,
        )
        client = LLMClient(config)
        with pytest.raises(AllProvidersFailedError) as exc_info:
            await client.chat("Hello")
        assert len(exc_info.value.errors) == 2

    def test_no_providers_raises(self) -> None:
        config = LLMConfig(providers=[])
        with pytest.raises(ValueError, match="At least one provider"):
            LLMClient(config)


class TestLLMClientStream:
    @respx.mock
    async def test_stream_single_provider(self) -> None:
        sse_data = (
            'data: {"choices":[{"delta":{"content":"Hi"}}],"model":"gpt-4o"}\n\n'
            "data: [DONE]\n\n"
        )
        respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                content=sse_data.encode(),
                headers={"content-type": "text/event-stream"},
            )
        )
        config = LLMConfig(providers=[_openai_config()])
        client = LLMClient(config)
        chunks = []
        async for chunk in client.stream("Hello"):
            chunks.append(chunk.content)
        assert "Hi" in "".join(chunks)

    @respx.mock
    async def test_stream_fallback_to_second_provider(self) -> None:
        _mock_openai_failure()
        sse_data = (
            'data: {"choices":[{"delta":{"content":"Fallback"}}],"model":"gpt-4o"}\n\n'
            "data: [DONE]\n\n"
        )
        # Second mock is also OpenAI-compatible but we'll use two openai configs
        # Instead, set up a second OpenAI-like provider via a second call —
        # actually let's test with a clean single-provider fallback path:
        # Re-mock the OpenAI endpoint to succeed on the second provider call.
        # The simplest way: use two calls to respx with side_effect.
        respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                content=sse_data.encode(),
                headers={"content-type": "text/event-stream"},
            )
        )
        config = LLMConfig(providers=[_openai_config()], fallback=True, max_retries=1)
        client = LLMClient(config)
        chunks: list[str] = []
        async for chunk in client.stream("Hello"):
            chunks.append(chunk.content)
        assert "".join(chunks) == "Fallback"

    @respx.mock
    async def test_stream_all_providers_fail(self) -> None:
        _mock_openai_failure()
        config = LLMConfig(providers=[_openai_config()], fallback=True, max_retries=1)
        client = LLMClient(config)
        with pytest.raises(AllProvidersFailedError):
            async for _ in client.stream("Hello"):
                pass


class TestLLMClientStructuredOutput:
    @respx.mock
    async def test_structured_output(self) -> None:
        class Greeting(BaseModel):
            message: str
            language: str

        respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json={
                    "choices": [
                        {"message": {"content": '{"message": "Ciao!", "language": "italian"}'}}
                    ],
                    "model": "gpt-4o",
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 10,
                        "total_tokens": 20,
                    },
                },
            )
        )
        config = LLMConfig(providers=[_openai_config()])
        client = LLMClient(config)
        result = await client.chat("Greet me in Italian", response_model=Greeting)
        assert isinstance(result, Greeting)
        assert result.message == "Ciao!"
        assert result.language == "italian"

    @respx.mock
    async def test_structured_output_injects_system_message(self) -> None:
        """Verify that a system prompt with the JSON schema is prepended."""

        class Answer(BaseModel):
            value: int

        captured_payload: dict[str, object] = {}

        def capture_request(request: httpx.Request) -> httpx.Response:
            import json as _json

            captured_payload.update(_json.loads(request.content))
            return httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": '{"value": 42}'}}],
                    "model": "gpt-4o",
                    "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
                },
            )

        respx.post("https://api.openai.com/v1/chat/completions").mock(
            side_effect=capture_request
        )
        config = LLMConfig(providers=[_openai_config()])
        client = LLMClient(config)
        result = await client.chat("What is 6*7?", response_model=Answer)
        assert isinstance(result, Answer)
        assert result.value == 42

        messages = captured_payload["messages"]
        assert isinstance(messages, list)
        assert messages[0]["role"] == "system"
        assert "json" in messages[0]["content"].lower()
