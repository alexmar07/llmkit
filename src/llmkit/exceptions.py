"""LLMKit exceptions."""
from __future__ import annotations


class LLMKitError(Exception):
    """Base exception for LLMKit."""


class ProviderError(LLMKitError):
    """Raised when a single provider fails."""

    def __init__(self, provider: str, message: str) -> None:
        self.provider = provider
        super().__init__(f"[{provider}] {message}")


class AllProvidersFailedError(LLMKitError):
    """Raised when all providers fail during fallback."""

    def __init__(self, errors: list[ProviderError]) -> None:
        self.errors = errors
        providers = ", ".join(e.provider for e in errors)
        super().__init__(f"All providers failed: {providers}")
