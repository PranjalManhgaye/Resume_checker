"""Tests for Gemini client utilities."""

from __future__ import annotations

from services.gemini_client import _error_status_code, _mask_api_key, _user_message_for_status
from services.llm_client import LLMAPIError


def test_mask_api_key() -> None:
    assert _mask_api_key("abcdefghij") == "****ghij"


def test_user_message_for_503() -> None:
    msg = _user_message_for_status(503)
    assert "overloaded" in msg.lower() or "wait" in msg.lower()


def test_error_status_code_from_api_error() -> None:
    class FakeAPIError(Exception):
        code = 429

    assert _error_status_code(FakeAPIError()) == 429


def test_gemini_api_error_has_user_message() -> None:
    err = LLMAPIError("internal", user_message="Try again later.", status_code=503)
    assert err.user_message == "Try again later."
    assert err.status_code == 503
