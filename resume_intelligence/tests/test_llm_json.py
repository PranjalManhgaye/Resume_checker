"""Tests for LLM JSON parsing utility."""

from __future__ import annotations

import pytest

from utils.llm_json import LLMJSONError, parse_llm_json


def test_parse_plain_json_object() -> None:
    assert parse_llm_json('{"a": 1}') == {"a": 1}


def test_parse_json_fence() -> None:
    text = 'Here you go:\n```json\n{"x": "y"}\n```'
    assert parse_llm_json(text) == {"x": "y"}


def test_parse_embedded_object() -> None:
    text = 'Sure! {"original": "a", "improved": "b"} hope that helps'
    assert parse_llm_json(text)["improved"] == "b"


def test_empty_raises() -> None:
    with pytest.raises(LLMJSONError):
        parse_llm_json("")
