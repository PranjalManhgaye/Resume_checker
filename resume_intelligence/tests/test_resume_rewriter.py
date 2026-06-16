"""Tests for resume bullet improvement."""

from __future__ import annotations

from unittest.mock import MagicMock

from backend.resume_rewriter import improve_resume
from models.resume import ResumeData


def test_improve_single_bullet_via_json() -> None:
    resume = ResumeData(
        projects=[
            {
                "title": "Budget App",
                "description": "Built a C++ console app for expense tracking",
            }
        ],
    )
    mock = MagicMock()
    mock.generate_json.return_value = {
        "original": "Built a C++ console app for expense tracking",
        "improved": "Developed a C++ console application for expense tracking",
    }

    result = improve_resume(resume, llm_client=mock)
    assert len(result.projects) == 1
    assert result.projects[0].unchanged is False
    assert "Developed" in result.projects[0].improved


def test_no_bullets_returns_error() -> None:
    result = improve_resume(ResumeData(), llm_client=MagicMock())
    assert result.errors
    assert not result.experience and not result.projects
