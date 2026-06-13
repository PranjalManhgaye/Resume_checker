"""Golden-output regression tests for LLM-backed features (no API keys required)."""

from __future__ import annotations

import json

import pytest

from backend.candidate_summary import generate_candidate_summary
from backend.form_filler import FormQuestion, fill_form
from backend.resume_rewriter import improve_resume
from models.resume import ResumeData
from llm_fixtures import (
    BANNED_LLM_PHRASES,
    FixtureLLMClient,
    load_llm_json_fixture,
)


def test_form_fill_golden_answers(
    sample_resume_alex: ResumeData,
    form_fill_mock_client: FixtureLLMClient,
) -> None:
    """Form fill with mocked LLM must match golden expected answers."""
    questions_json = load_llm_json_fixture("form_fill_questions.json")
    expected = load_llm_json_fixture("form_fill_expected_answers.json")

    questions = [FormQuestion(q) for q in questions_json]
    answers = fill_form(sample_resume_alex, questions, llm_client=form_fill_mock_client)

    assert answers == expected
    assert form_fill_mock_client.prompts, "LLM client should be called for open-ended questions"
    assert "Alex Johnson" in form_fill_mock_client.prompts[0]


def test_rewriter_golden_output_differs_and_avoids_banned_phrases(
    sample_resume_alex: ResumeData,
    rewriter_mock_client: FixtureLLMClient,
) -> None:
    """Rewriter output must differ from input and avoid banned AI phrases."""
    golden = load_llm_json_fixture("rewriter_response.json")
    result = improve_resume(sample_resume_alex, llm_client=rewriter_mock_client)

    assert not result.errors
    assert rewriter_mock_client.prompts

    all_bullets = result.experience + result.projects
    assert len(all_bullets) == len(golden["experience"]) + len(golden["projects"])

    for bullet in all_bullets:
        assert bullet.improved.strip(), "Improved bullet must not be empty"
        assert bullet.original.strip() != bullet.improved.strip(), (
            f"Improved bullet should differ from original: {bullet.original!r}"
        )
        improved_lower = bullet.improved.lower()
        for phrase in BANNED_LLM_PHRASES:
            assert phrase not in improved_lower, f"Banned phrase {phrase!r} in: {bullet.improved!r}"

    for section, key in (("experience", "experience"), ("projects", "projects")):
        expected_items = golden[key]
        actual_items = getattr(result, section)
        for expected, actual in zip(expected_items, actual_items, strict=True):
            assert actual.original == expected["original"]
            assert actual.improved == expected["improved"]


def test_summary_golden_output(
    sample_resume_alex: ResumeData,
    summary_mock_client: FixtureLLMClient,
) -> None:
    """Candidate summary with mocked LLM must match golden text."""
    expected = summary_mock_client.response.strip()
    summary = generate_candidate_summary(
        sample_resume_alex,
        job_description="Backend engineer role requiring FastAPI and Docker.",
        llm_client=summary_mock_client,
    )

    assert summary.strip() == expected
    assert summary_mock_client.prompts
    assert "Alex Johnson" in summary_mock_client.prompts[0]


def test_form_fill_mock_client_returns_valid_json(form_fill_mock_client: FixtureLLMClient) -> None:
    """Golden form-fill fixture must be valid JSON mapping questions to answers."""
    parsed = json.loads(form_fill_mock_client.response)
    assert isinstance(parsed, dict)
    assert len(parsed) == 3


def test_rewriter_mock_client_returns_valid_json(rewriter_mock_client: FixtureLLMClient) -> None:
    """Golden rewriter fixture must contain experience and projects arrays."""
    parsed = json.loads(rewriter_mock_client.response)
    assert "experience" in parsed
    assert "projects" in parsed
    assert all("original" in item and "improved" in item for item in parsed["experience"])
    assert all("original" in item and "improved" in item for item in parsed["projects"])
