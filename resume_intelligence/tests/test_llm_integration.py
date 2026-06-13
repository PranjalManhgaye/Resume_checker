"""Live LLM API smoke tests — skipped in CI unless API keys are configured."""

from __future__ import annotations

import os

import pytest

from services.llm_client import get_llm_client, reset_llm_client


def _has_live_llm_credentials() -> bool:
    provider = os.getenv("LLM_PROVIDER", "groq").strip().lower()
    if provider == "groq":
        return bool(os.getenv("GROQ_API_KEY"))
    return bool(os.getenv("GEMINI_API_KEY"))


@pytest.mark.integration
def test_live_llm_generate_text() -> None:
    """Smoke test against the configured LLM provider (manual / optional CI)."""
    if not _has_live_llm_credentials():
        pytest.skip("Set GROQ_API_KEY or GEMINI_API_KEY to run live LLM tests")

    reset_llm_client()
    client = get_llm_client()
    response = client.generate_text("Reply with exactly: ok", max_output_tokens=16)

    assert response.strip()
    assert "as an ai" not in response.lower()
