"""Tests for TF-IDF scoring utilities."""

from __future__ import annotations

from backend.tfidf_scorer import extract_jd_keywords, keyword_alignment_score, tfidf_document_similarity


def test_extract_jd_keywords() -> None:
    jd = "Python developer with FastAPI, Docker, PostgreSQL, and REST API experience."
    keywords = extract_jd_keywords(jd, top_n=10)
    assert len(keywords) > 0
    joined = " ".join(keywords).lower()
    assert "python" in joined or "fastapi" in joined or "developer" in joined


def test_tfidf_document_similarity() -> None:
    resume = "Python FastAPI Docker developer"
    jd = "Looking for Python FastAPI engineer with Docker experience"
    sim = tfidf_document_similarity(resume, jd)
    assert 0 < sim <= 1.0


def test_keyword_alignment_score() -> None:
    resume = "Experienced in Python, FastAPI, Docker, and PostgreSQL databases."
    jd = "Python FastAPI Docker PostgreSQL REST APIs required"
    score, breakdown = keyword_alignment_score(resume, jd)
    assert score > 0
    assert len(breakdown["matched_keywords"]) > 0
