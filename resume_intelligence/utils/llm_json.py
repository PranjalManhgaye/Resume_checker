"""Robust JSON extraction from LLM text responses."""

from __future__ import annotations

import json
import re
from typing import Any

from utils.logger import get_logger

logger = get_logger(__name__)

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)
_JSON_OBJECT_RE = re.compile(r"\{[\s\S]*\}")
_JSON_ARRAY_RE = re.compile(r"\[[\s\S]*\]")


class LLMJSONError(ValueError):
    """Raised when LLM output cannot be parsed as JSON."""


def parse_llm_json(text: str) -> Any:
    """
    Parse JSON from LLM output — handles fences, prose wrappers, and truncation.

    Raises LLMJSONError when no valid JSON is found.
    """
    if not text or not text.strip():
        raise LLMJSONError("LLM returned an empty response.")

    cleaned = text.strip()

    # Plain JSON
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # ```json ... ``` block
    fence_match = _JSON_FENCE_RE.search(cleaned)
    if fence_match:
        try:
            return json.loads(fence_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # First object or array embedded in prose
    for pattern in (_JSON_OBJECT_RE, _JSON_ARRAY_RE):
        match = pattern.search(cleaned)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                continue

    logger.warning("Failed to parse LLM JSON (first 200 chars): %s", cleaned[:200])
    raise LLMJSONError("Could not parse AI response as JSON. Try again in a moment.")
