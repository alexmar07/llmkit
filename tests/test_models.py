"""Tests for LLMKit data models."""
from llmkit.models import ChatResponse, Message, StreamChunk, Usage


class TestMessage:
    def test_user_message(self) -> None:
        msg = Message(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_assistant_message(self) -> None:
        msg = Message(role="assistant", content="Hi there")
        assert msg.role == "assistant"

    def test_system_message(self) -> None:
        msg = Message(role="system", content="You are helpful")
        assert msg.role == "system"

    def test_invalid_role_rejected(self) -> None:
        import pytest
        with pytest.raises(ValueError):
            Message(role="invalid", content="test")

class TestUsage:
    def test_usage_fields(self) -> None:
        usage = Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        assert usage.prompt_tokens == 10
        assert usage.completion_tokens == 20
        assert usage.total_tokens == 30

    def test_usage_defaults_to_zero(self) -> None:
        usage = Usage()
        assert usage.prompt_tokens == 0
        assert usage.total_tokens == 0

class TestChatResponse:
    def test_chat_response_fields(self) -> None:
        response = ChatResponse(
            content="Hello!", provider="openai", model="gpt-4o",
            usage=Usage(prompt_tokens=5, completion_tokens=3, total_tokens=8),
        )
        assert response.content == "Hello!"
        assert response.provider == "openai"
        assert response.model == "gpt-4o"
        assert response.usage.total_tokens == 8

    def test_chat_response_optional_usage(self) -> None:
        response = ChatResponse(content="Hi", provider="ollama", model="llama3")
        assert response.usage is None

class TestStreamChunk:
    def test_stream_chunk(self) -> None:
        chunk = StreamChunk(content="Hel", provider="openai", model="gpt-4o")
        assert chunk.content == "Hel"

    def test_stream_chunk_done(self) -> None:
        chunk = StreamChunk(
            content="", provider="openai", model="gpt-4o", done=True,
            usage=Usage(prompt_tokens=5, completion_tokens=10, total_tokens=15),
        )
        assert chunk.done is True
        assert chunk.usage is not None
