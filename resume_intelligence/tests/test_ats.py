"""Tests for ATS scoring engine."""

from __future__ import annotations

from backend.ats import compute_ats_score
from models.resume import ResumeData

SAMPLE_JD = """
We are looking for a Python developer with experience in FastAPI, Docker,
PostgreSQL, and REST APIs. Knowledge of machine learning is a plus.
"""


def test_ats_score_returns_breakdown() -> None:
    resume = ResumeData(
        name="Alex Johnson",
        email="alex@email.com",
        skills=["Python", "FastAPI", "Docker", "SQL", "React"],
        experience=[
            {
                "title": "Software Intern",
                "org": "TechCorp",
                "dates": "2022",
                "description": "Built REST APIs using FastAPI and PostgreSQL",
            }
        ],
        raw_text="Python developer with FastAPI and Docker experience.",
    )

    result = compute_ats_score(resume, SAMPLE_JD)

    assert 0 <= result.ats_score <= 100
    assert result.skill_match_score > 0
    assert "matched_skills" in result.breakdown


def test_ats_score_empty_jd() -> None:
    resume = ResumeData(skills=["Python"])
    result = compute_ats_score(resume, "")
    assert result.ats_score == 0.0


def test_ats_score_weights_applied() -> None:
    resume = ResumeData(
        skills=["Python", "FastAPI", "Docker", "PostgreSQL"],
        raw_text="Python FastAPI Docker PostgreSQL REST API developer",
        experience=[
            {"title": "Dev", "org": "Co", "dates": "2022", "description": "FastAPI PostgreSQL"}
        ],
    )

    result = compute_ats_score(resume, SAMPLE_JD)
    assert result.ats_score > 30
