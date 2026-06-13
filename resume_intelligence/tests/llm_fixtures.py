"""Shared helpers and constants for LLM golden tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from utils.llm_json import parse_llm_json

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

    def generate_json(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_output_tokens: Optional[int] = None,
    ) -> Any:
        del system, max_output_tokens
        self.prompts.append(prompt)
        return parse_llm_json(self.response)


class RewriterFixtureLLMClient(FixtureLLMClient):
    """Mock rewriter client — returns one bullet JSON per call (matches #26 per-bullet API)."""

    def __init__(self, response: str) -> None:
        super().__init__(response)
        golden = json.loads(response)
        self._bullets: list[dict[str, str]] = []
        for section in ("experience", "projects"):
            for item in golden.get(section, []):
                self._bullets.append(item)

    def generate_json(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_output_tokens: Optional[int] = None,
    ) -> Any:
        del system, max_output_tokens
        self.prompts.append(prompt)
        for item in self._bullets:
            original = item.get("original", "")
            if original and original in prompt:
                return item
        idx = len(self.prompts) - 1
        if 0 <= idx < len(self._bullets):
            return self._bullets[idx]
        return {"original": "unknown", "improved": "unknown"}


def load_llm_fixture(name: str) -> str:
    """Load a golden fixture file from tests/fixtures/llm/."""
    path = FIXTURES_DIR / name
    return path.read_text(encoding="utf-8")


def load_llm_json_fixture(name: str) -> dict | list:
    """Load and parse a JSON golden fixture."""
    return json.loads(load_llm_fixture(name))
