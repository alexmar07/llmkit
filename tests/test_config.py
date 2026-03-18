"""Tests for LLMWire configuration."""
import os
from unittest.mock import patch

from llmwire.config import LLMConfig, ProviderConfig


class TestProviderConfig:
    def test_provider_config_fields(self) -> None:
        config = ProviderConfig(name="openai", api_key="sk-test", model="gpt-4o")
        assert config.name == "openai"
        assert config.api_key == "sk-test"
        assert config.model == "gpt-4o"

    def test_provider_config_optional_base_url(self) -> None:
        config = ProviderConfig(name="ollama", model="llama3")
        assert config.api_key is None
        assert config.base_url is None

    def test_provider_config_custom_base_url(self) -> None:
        config = ProviderConfig(name="ollama", model="llama3", base_url="http://localhost:11434")
        assert config.base_url == "http://localhost:11434"


class TestLLMConfig:
    def test_config_with_providers(self) -> None:
        provider = ProviderConfig(name="openai", api_key="sk-test", model="gpt-4o")
        config = LLMConfig(providers=[provider], fallback=False)
        assert len(config.providers) == 1
        assert config.fallback is False

    def test_config_defaults(self) -> None:
        provider = ProviderConfig(name="openai", api_key="sk-test", model="gpt-4o")
        config = LLMConfig(providers=[provider])
        assert config.fallback is True
        assert config.max_retries == 3
        assert config.timeout == 30.0

    def test_config_from_env(self) -> None:
        env = {
            "LLMKIT_PROVIDERS__0__NAME": "openai",
            "LLMKIT_PROVIDERS__0__API_KEY": "sk-env-test",
            "LLMKIT_PROVIDERS__0__MODEL": "gpt-4o-mini",
        }
        with patch.dict(os.environ, env, clear=False):
            config = LLMConfig()  # type: ignore[call-arg]
            assert len(config.providers) >= 1
