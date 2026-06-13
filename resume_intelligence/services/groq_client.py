"""Groq API client for fast text generation.

Groq API docs: https://console.groq.com/docs/quickstart
"""

from __future__ import annotations

import os
import time
from typing import Any, Optional

from dotenv import load_dotenv
from groq import Groq

from services.llm_client import ConfigurationError, LLMAPIError, MISSING_INFO
from utils.llm_json import LLMJSONError, parse_llm_json
from utils.logger import get_logger

load_dotenv()

logger = get_logger(__name__)

RETRYABLE_STATUS_CODES = {429, 503}
MAX_RETRIES = 3
RETRY_BASE_DELAY_SEC = 1.0


def _mask_api_key(key: str) -> str:
    if len(key) <= 4:
        return "****"
    return f"****{key[-4:]}"


def _parse_fallback_models() -> list[str]:
    raw = os.getenv("GROQ_MODEL_FALLBACK", "").strip()
    if not raw:
        return []
    return [m.strip() for m in raw.split(",") if m.strip()]


def _user_message_for_status(status_code: Optional[int]) -> str:
    if status_code == 503:
        return "Groq is temporarily overloaded. Please wait a moment and try again."
    if status_code == 429:
        return "Groq rate limit reached. Please wait a minute and try again."
    if status_code == 403:
        return "API access denied. Check your GROQ_API_KEY in .env."
    return "AI request failed. Please try again."


class GroqClient:
    """Thin wrapper around the Groq SDK with retry and model fallback."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None) -> None:
        self.api_key = api_key or os.getenv("GROQ_API_KEY", "").strip()
        self.model = model or os.getenv("GROQ_MODEL", "llama-3.1-8b-instant").strip()
        self.fallback_models = _parse_fallback_models()

        if not self.api_key:
            raise ConfigurationError(
                "GROQ_API_KEY is not set. Add it to your .env file or set LLM_PROVIDER=gemini."
            )

        self._client = Groq(api_key=self.api_key)
        logger.info(
            "Groq client initialized | model=%s | key=%s",
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
        """Generate text using Groq chat completions."""
        if not prompt.strip():
            return MISSING_INFO

        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        last_error: Optional[Exception] = None
        last_status: Optional[int] = None

        for model_name in self._models_to_try():
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    kwargs: dict = {"model": model_name, "messages": messages}
                    if max_output_tokens:
                        kwargs["max_tokens"] = max_output_tokens

                    response = self._client.chat.completions.create(**kwargs)
                    text = (response.choices[0].message.content or "").strip()
                    return text if text else MISSING_INFO

                except Exception as exc:
                    last_error = exc
                    last_status = getattr(exc, "status_code", None)
                    logger.warning(
                        "Groq attempt %d/%d failed | model=%s | status=%s",
                        attempt,
                        MAX_RETRIES,
                        model_name,
                        last_status,
                    )
                    if last_status in RETRYABLE_STATUS_CODES and attempt < MAX_RETRIES:
                        delay = RETRY_BASE_DELAY_SEC * (2 ** (attempt - 1))
                        logger.info("Retrying in %.1fs...", delay)
                        time.sleep(delay)
                        continue
                    break

        raise LLMAPIError(
            str(last_error),
            user_message=_user_message_for_status(last_status),
            status_code=last_status,
        )

    def generate_json(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_output_tokens: Optional[int] = None,
    ) -> Any:
        """Generate and parse JSON — uses Groq json_object mode when available."""
        json_system = (system or "") + "\nYou must respond with valid JSON only."
        messages: list[dict[str, str]] = []
        if json_system.strip():
            messages.append({"role": "system", "content": json_system.strip()})
        messages.append({"role": "user", "content": prompt})

        last_error: Optional[Exception] = None
        last_status: Optional[int] = None

        for model_name in self._models_to_try():
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    kwargs: dict = {
                        "model": model_name,
                        "messages": messages,
                        "response_format": {"type": "json_object"},
                    }
                    if max_output_tokens:
                        kwargs["max_tokens"] = max_output_tokens

                    response = self._client.chat.completions.create(**kwargs)
                    text = (response.choices[0].message.content or "").strip()
                    return parse_llm_json(text)

                except LLMJSONError as exc:
                    last_error = exc
                    logger.warning("Groq JSON parse failed for model=%s: %s", model_name, exc)
                    break
                except Exception as exc:
                    last_error = exc
                    last_status = getattr(exc, "status_code", None)
                    # Some models reject response_format — fall back to text parse once
                    if "response_format" in str(exc).lower() or last_status == 400:
                        try:
                            text = self.generate_text(prompt, system=json_system, max_output_tokens=max_output_tokens)
                            return parse_llm_json(text)
                        except (LLMJSONError, LLMAPIError) as inner:
                            last_error = inner
                            break
                    if last_status in RETRYABLE_STATUS_CODES and attempt < MAX_RETRIES:
                        time.sleep(RETRY_BASE_DELAY_SEC * (2 ** (attempt - 1)))
                        continue
                    break

        if isinstance(last_error, LLMJSONError):
            raise LLMAPIError(str(last_error), user_message=str(last_error))
        raise LLMAPIError(
            str(last_error),
            user_message=_user_message_for_status(last_status),
            status_code=last_status,
        )


_client_instance: Optional[GroqClient] = None


def reset_groq_client() -> None:
    global _client_instance
    _client_instance = None


def get_groq_client() -> GroqClient:
    global _client_instance
    if _client_instance is None:
        _client_instance = GroqClient()
    return _client_instance
