"""Gemini API client for text generation features.

Based on Gemini API quickstart:
https://ai.google.dev/gemini-api/docs/quickstart
"""

from __future__ import annotations

import os
import time
from typing import Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import ClientError

from services.llm_client import ConfigurationError, LLMAPIError, MISSING_INFO
from utils.logger import get_logger

load_dotenv()

logger = get_logger(__name__)

# Backward-compatible alias
GeminiAPIError = LLMAPIError

RETRYABLE_STATUS_CODES = {429, 503}
SKIP_MODEL_STATUS_CODES = {404}  # model name invalid — try next fallback immediately
MAX_RETRIES = 3
RETRY_BASE_DELAY_SEC = 2.0


def _mask_api_key(key: str) -> str:
    if len(key) <= 4:
        return "****"
    return f"****{key[-4:]}"


def _parse_fallback_models() -> list[str]:
    raw = os.getenv("GEMINI_MODEL_FALLBACK", "").strip()
    if not raw:
        return []
    return [m.strip() for m in raw.split(",") if m.strip()]


def _error_status_code(exc: Exception) -> Optional[int]:
    """Extract HTTP status from google-genai APIError (uses .code, not .status_code)."""
    code = getattr(exc, "code", None)
    if isinstance(code, int):
        return code
    return getattr(exc, "status_code", None)


def _user_message_for_status(status_code: Optional[int]) -> str:
    if status_code == 503:
        return "Gemini is temporarily overloaded. Please wait a moment and try again."
    if status_code == 429:
        return "Gemini quota exceeded. Wait a few minutes or check your API plan at Google AI Studio."
    if status_code == 403:
        return "API access denied. Check your GEMINI_API_KEY in .env."
    if status_code == 404:
        return "Configured Gemini model not available. Update GEMINI_MODEL in .env."
    return "AI request failed. Please try again."


class GeminiClient:
    """Thin wrapper around the google-genai SDK with retry and model fallback."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None) -> None:
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "").strip()
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-2.0-flash").strip()
        self.fallback_models = _parse_fallback_models()

        if not self.api_key:
            raise ConfigurationError(
                "GEMINI_API_KEY is not set. Add it to your .env file before using AI features."
            )

        self._client = genai.Client(api_key=self.api_key)
        logger.info(
            "Gemini client initialized | model=%s | key=%s",
            self.model,
            _mask_api_key(self.api_key),
        )

    def _models_to_try(self) -> list[str]:
        models = [self.model]
        for fallback in self.fallback_models:
            if fallback not in models:
                models.append(fallback)
        return models

    def generate_text(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_output_tokens: Optional[int] = None,
    ) -> str:
        """
        Generate text from a prompt with retry and model fallback.

        Args:
            prompt: User/task prompt.
            system: Optional system instruction for behavior constraints.
            max_output_tokens: Cap response length for faster replies.
        """
        if not prompt.strip():
            return MISSING_INFO

        last_error: Optional[Exception] = None
        last_status: Optional[int] = None

        for model_name in self._models_to_try():
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    config_kwargs: dict = {}
                    if system:
                        config_kwargs["system_instruction"] = system
                    if max_output_tokens:
                        config_kwargs["max_output_tokens"] = max_output_tokens

                    config = types.GenerateContentConfig(**config_kwargs) if config_kwargs else None

                    response = self._client.models.generate_content(
                        model=model_name,
                        contents=prompt,
                        config=config,
                    )
                    text = (response.text or "").strip()
                    return text if text else MISSING_INFO

                except ClientError as exc:
                    last_error = exc
                    last_status = _error_status_code(exc)
                    logger.warning(
                        "Gemini attempt %d/%d failed | model=%s | status=%s",
                        attempt,
                        MAX_RETRIES,
                        model_name,
                        last_status,
                    )
                    # Bad model name — skip retries and try the next fallback model
                    if last_status in SKIP_MODEL_STATUS_CODES:
                        logger.info("Model %s not available, trying next fallback.", model_name)
                        break
                    if last_status in RETRYABLE_STATUS_CODES and attempt < MAX_RETRIES:
                        delay = RETRY_BASE_DELAY_SEC * (2 ** (attempt - 1))
                        logger.info("Retrying in %.1fs...", delay)
                        time.sleep(delay)
                        continue
                    break

                except Exception as exc:
                    last_error = exc
                    logger.error("Gemini API call failed: %s", exc)
                    break

        raise LLMAPIError(
            str(last_error),
            user_message=_user_message_for_status(last_status),
            status_code=last_status,
        )


_client_instance: Optional[GeminiClient] = None


def reset_gemini_client() -> None:
    """Clear cached client so .env changes take effect after restart."""
    global _client_instance
    _client_instance = None


def get_gemini_client() -> GeminiClient:
    """Return a shared GeminiClient instance."""
    global _client_instance
    if _client_instance is None:
        _client_instance = GeminiClient()
    return _client_instance
