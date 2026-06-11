"""Unified LLM client factory — routes to Gemini or Groq based on LLM_PROVIDER."""

from __future__ import annotations

import os
from typing import Optional, Protocol, runtime_checkable

from dotenv import load_dotenv

load_dotenv()

MISSING_INFO = "Information not available."


class ConfigurationError(Exception):
    """Raised when required environment configuration is missing."""


class LLMAPIError(Exception):
    """Raised when LLM API calls fail after retries."""

    def __init__(self, message: str, user_message: str = "", status_code: Optional[int] = None) -> None:
        super().__init__(message)
        self.user_message = user_message or message
        self.status_code = status_code


# Backward-compatible alias used across the codebase
GeminiAPIError = LLMAPIError


@runtime_checkable
class LLMClient(Protocol):
    """Common interface for text-generation providers."""

    def generate_text(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_output_tokens: Optional[int] = None,
    ) -> str: ...


_client_instance: Optional[LLMClient] = None


def reset_llm_client() -> None:
    """Clear cached client so .env changes take effect after restart."""
    global _client_instance
    _client_instance = None
    from services.gemini_client import reset_gemini_client
    from services.groq_client import reset_groq_client

    reset_gemini_client()
    reset_groq_client()


def get_llm_client() -> LLMClient:
    """Return the configured LLM client (gemini or groq)."""
    global _client_instance
    if _client_instance is not None:
        return _client_instance

    provider = os.getenv("LLM_PROVIDER", "gemini").strip().lower()

    if provider == "groq":
        from services.groq_client import get_groq_client

        _client_instance = get_groq_client()
    else:
        from services.gemini_client import get_gemini_client

        _client_instance = get_gemini_client()

    return _client_instance


def get_gemini_client() -> LLMClient:
    """Backward-compatible alias — returns active LLM client."""
    return get_llm_client()
