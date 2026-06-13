"""Shared helpers and constants for LLM golden tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures" / "llm"

BANNED_LLM_PHRASES = (
    "as an ai",
    "as a language model",
    "i cannot",
    "i'm an ai",
)


class FixtureLLMClient:
    """Mock LLM client that returns a fixed response for golden tests."""

    def __init__(self, response: str) -> None:
        self.response = response
        self.prompts: list[str] = []

    def generate_text(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_output_tokens: Optional[int] = None,
    ) -> str:
        del system, max_output_tokens
        self.prompts.append(prompt)
        return self.response


def load_llm_fixture(name: str) -> str:
    """Load a golden fixture file from tests/fixtures/llm/."""
    path = FIXTURES_DIR / name
    return path.read_text(encoding="utf-8")


def load_llm_json_fixture(name: str) -> dict | list:
    """Load and parse a JSON golden fixture."""
    return json.loads(load_llm_fixture(name))
