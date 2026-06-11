"""Tests for application form autofill."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from backend.form_filler import FormQuestion, fill_form
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


def test_gemini_called_once_for_batch() -> None:
    resume = ResumeData(
        name="Alex",
        experience=[
            {"title": "Intern", "org": "TechCorp", "dates": "2022", "description": "Built APIs"}
        ],
    )

    mock_client = MagicMock()
    mock_client.generate_text.return_value = json.dumps({
        "Describe your work experience.": "Built REST APIs at TechCorp.",
        "Why do you want this role?": "Interested in backend development.",
    })

    questions = [
        FormQuestion("Describe your work experience."),
        FormQuestion("Why do you want this role?"),
    ]
    answers = fill_form(resume, questions, llm_client=mock_client)

    mock_client.generate_text.assert_called_once()
    assert "Built REST APIs" in answers["Describe your work experience."]


def test_missing_data_returns_not_available() -> None:
    resume = ResumeData()
    questions = [FormQuestion("What is your full name?")]
    answers = fill_form(resume, questions, llm_client=MagicMock())
    assert answers["What is your full name?"] == MISSING_INFO
