"""Tests for ATS scoring engine."""

from __future__ import annotations

from backend.ats import DIMENSION_NAMES, compute_ats_score
from models.resume import ResumeData

SAMPLE_JD = """
We are looking for a Python developer with experience in FastAPI, Docker,
PostgreSQL, and REST APIs. Knowledge of machine learning is a plus.
Must improve system performance and deliver scalable solutions.
"""


def test_ats_score_returns_eight_dimensions() -> None:
    resume = ResumeData(
        name="Alex Johnson",
        email="alex@email.com",
        skills=["Python", "FastAPI", "Docker", "SQL", "React"],
        education=[{"title": "B.Tech", "org": "University", "dates": "2023", "description": ""}],
        projects=[{"title": "API Project", "org": "", "dates": "2022", "description": "Built REST APIs"}],
        experience=[
            {
                "title": "Software Intern",
                "org": "TechCorp",
                "dates": "2022",
                "description": "Built and optimized REST APIs using FastAPI, improved latency by 30%",
            }
        ],
        raw_text="Python developer with FastAPI and Docker experience. Built REST APIs.",
    )

    result = compute_ats_score(resume, SAMPLE_JD)

    assert 0 <= result.ats_score <= 100
    assert len(result.dimensions) == 8
    for name in DIMENSION_NAMES:
        assert name in result.dimensions
    assert "tfidf_keywords" in result.breakdown
    assert "matched_skills" in result.breakdown


def test_ats_score_empty_jd() -> None:
    resume = ResumeData(skills=["Python"])
    result = compute_ats_score(resume, "")
    assert result.ats_score == 0.0


def test_ats_score_strong_resume_scores_higher() -> None:
    weak = ResumeData(skills=["Python"], raw_text="python")
    strong = ResumeData(
        name="Alex",
        email="alex@email.com",
        skills=["Python", "FastAPI", "Docker", "PostgreSQL"],
        education=[{"title": "B.Tech", "org": "Uni", "dates": "2023", "description": ""}],
        experience=[
            {
                "title": "Dev",
                "org": "Co",
                "dates": "2022",
                "description": "Developed FastAPI services, improved throughput by 40%, led deployment",
            }
        ],
        projects=[{"title": "App", "org": "", "dates": "2021", "description": "Built Dockerized API"}],
        raw_text="Python FastAPI Docker PostgreSQL REST API developer improved performance by 40%",
    )

    weak_result = compute_ats_score(weak, SAMPLE_JD)
    strong_result = compute_ats_score(strong, SAMPLE_JD)
    assert strong_result.ats_score > weak_result.ats_score


def test_tfidf_keyword_alignment() -> None:
    resume = ResumeData(
        skills=["Python", "FastAPI"],
        raw_text="Python FastAPI Docker PostgreSQL REST API",
    )
    result = compute_ats_score(resume, SAMPLE_JD)
    assert result.dimensions["ats_keyword_alignment"] > 0
    assert result.dimensions["role_fit"] > 0
