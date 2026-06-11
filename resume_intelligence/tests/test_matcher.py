"""Tests for skill gap matcher."""

from __future__ import annotations

from backend.matcher import analyze_skill_gap, normalize_skill
from models.resume import ResumeData

SAMPLE_JD = """
Required: Python, JavaScript, React, Docker, AWS, PostgreSQL.
Nice to have: Kubernetes, machine learning.
"""


def test_normalize_skill_aliases() -> None:
    assert normalize_skill("JS") == "javascript"
    assert normalize_skill("k8s") == "kubernetes"


def test_skill_gap_matched_and_missing() -> None:
    resume = ResumeData(skills=["Python", "React", "Docker"])
    result = analyze_skill_gap(resume, SAMPLE_JD)

    assert "python" in result.matched_skills
    assert "javascript" in result.missing_skills
    assert result.match_percent > 0


def test_skill_gap_recommended_from_jd_only() -> None:
    resume = ResumeData(skills=["Python"])
    result = analyze_skill_gap(resume, SAMPLE_JD)

    for skill in result.recommended_skills:
        assert skill not in {normalize_skill(s) for s in resume.skills}


def test_skill_gap_empty_jd() -> None:
    resume = ResumeData(skills=["Python"])
    result = analyze_skill_gap(resume, "")
    assert result.match_percent == 0.0
