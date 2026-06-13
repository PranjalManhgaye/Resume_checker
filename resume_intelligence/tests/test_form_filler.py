"""Tests for application form autofill."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from backend.form_filler import FormQuestion, fill_form
from backend.parser import parse_resume
from models.resume import ResumeData
from services.llm_client import MISSING_INFO


def test_deterministic_name_and_email() -> None:
    resume = ResumeData(
        name="Alex Johnson",
        email="alex@email.com",
        education=[{"title": "B.Tech", "org": "State University", "dates": "2023", "description": "CGPA: 8.33"}],
        raw_text="CGPA: 8.33",
    )

    questions = [
        FormQuestion("What is your full name?"),
        FormQuestion("What is your email address?"),
        FormQuestion("What is your CGPA?"),
    ]

    answers = fill_form(resume, questions, llm_client=MagicMock())

    assert answers["What is your full name?"] == "Alex Johnson"
    assert answers["What is your email address?"] == "alex@email.com"
    assert answers["What is your CGPA?"] == "8.33"


def test_work_experience_from_parsed_sample_without_llm() -> None:
    sample = Path(__file__).resolve().parent.parent / "data/samples/sample_resume_well_formatted.pdf"
    result = parse_resume(sample.read_bytes(), sample.name)
    assert result.success and result.data

    questions = [FormQuestion("Describe your work experience.")]
    mock = MagicMock()
    answers = fill_form(result.data, questions, llm_client=mock)

    mock.generate_json.assert_not_called()
    answer = answers["Describe your work experience."]
    assert "FastAPI" in answer or "TechCorp" in answer
    assert answer != MISSING_INFO


def test_skills_from_resume_without_llm() -> None:
    resume = ResumeData(skills=["Python", "FastAPI", "Docker"])
    answers = fill_form(
        resume,
        [FormQuestion("List your technical skills.")],
        llm_client=MagicMock(),
    )
    assert "Python" in answers["List your technical skills."]


def test_llm_called_for_open_ended_when_no_resume_match() -> None:
    resume = ResumeData(name="Alex")
    mock_client = MagicMock()
    mock_client.generate_json.return_value = {
        "Why do you want this role?": "Interested in backend development.",
    }

    questions = [FormQuestion("Why do you want this role?")]
    answers = fill_form(resume, questions, llm_client=mock_client)

    mock_client.generate_json.assert_called_once()
    assert "backend" in answers["Why do you want this role?"].lower()


def test_fuzzy_json_key_matching() -> None:
    resume = ResumeData(name="Alex", raw_text="Some background info.")
    mock_client = MagicMock()
    mock_client.generate_json.return_value = {
        "Why do you want this role": "I enjoy building APIs.",
    }

    answers = fill_form(
        resume,
        [FormQuestion("Why do you want this role?")],
        llm_client=mock_client,
    )
    assert "API" in answers["Why do you want this role?"]


def test_missing_data_returns_not_available() -> None:
    resume = ResumeData()
    questions = [FormQuestion("What is your full name?")]
    answers = fill_form(resume, questions, llm_client=MagicMock())
    assert answers["What is your full name?"] == MISSING_INFO
