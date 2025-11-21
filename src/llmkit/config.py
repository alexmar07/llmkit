"""LLMKit configuration."""
from __future__ import annotations

from pydantic import BaseModel, field_validator
from pydantic_settings import BaseSettings


class ProviderConfig(BaseModel):
    """Configuration for a single LLM provider."""

    name: str
    api_key: str | None = None
    model: str = ""
    base_url: str | None = None

    @field_validator("name", mode="before")
    @classmethod
    def normalize_name(cls, v: object) -> object:
        """Normalize provider name to lowercase."""
        if isinstance(v, str):
            return v.strip().lower()
        return v


class LLMConfig(BaseSettings):
    """Main LLMKit configuration."""

    model_config = {"env_prefix": "LLMKIT_", "env_nested_delimiter": "__"}

    providers: list[ProviderConfig] = []
    fallback: bool = True
    max_retries: int = 3
    timeout: float = 30.0

    @field_validator("providers", mode="before")
    @classmethod
    def coerce_providers(cls, v: object) -> object:
        """Convert indexed-dict form (from env vars) to a list."""
        if isinstance(v, dict):
            return [v[k] for k in sorted(v.keys(), key=lambda x: int(x))]
        return v
