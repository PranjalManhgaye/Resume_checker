"""TF-IDF keyword extraction and similarity for ATS alignment.

Uses scikit-learn TfidfVectorizer for job-description keyword mapping:
https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
"""

from __future__ import annotations

import re
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from utils.logger import get_logger

logger = get_logger(__name__)


def tfidf_document_similarity(resume_text: str, job_description: str) -> float:
    """
    Cosine similarity between resume and job description TF-IDF vectors.

    Used for role-fit scoring — higher similarity indicates better alignment.
    """
    if not resume_text.strip() or not job_description.strip():
        return 0.0

    try:
        vectorizer = TfidfVectorizer(stop_words="english", max_features=3000, ngram_range=(1, 2))
        matrix = vectorizer.fit_transform([resume_text, job_description])
        sim = cosine_similarity(matrix[0:1], matrix[1:2])[0][0]
        return float(max(0.0, min(1.0, sim)))
    except Exception as exc:
        logger.warning("TF-IDF document similarity failed: %s", exc)
        return 0.0


def extract_jd_keywords(job_description: str, top_n: int = 25) -> list[str]:
    """Extract top TF-IDF terms from a job description."""
    if not job_description.strip():
        return []

    try:
        vectorizer = TfidfVectorizer(stop_words="english", max_features=500, ngram_range=(1, 2))
        matrix = vectorizer.fit_transform([job_description])
        features = vectorizer.get_feature_names_out()
        scores = matrix.toarray()[0]
        ranked = sorted(zip(features, scores), key=lambda x: x[1], reverse=True)
        return [term for term, score in ranked[:top_n] if score > 0]
    except Exception as exc:
        logger.warning("TF-IDF keyword extraction failed: %s", exc)
        return []


def keyword_alignment_score(resume_text: str, job_description: str) -> tuple[float, dict[str, Any]]:
    """
    Score how well resume text covers JD keywords extracted via TF-IDF.

    Returns score 0-100 and breakdown with matched/missing keywords.
    """
    keywords = extract_jd_keywords(job_description)
    if not keywords:
        return 0.0, {"matched_keywords": [], "missing_keywords": [], "jd_keywords": []}

    lower_resume = resume_text.lower()
    matched: list[str] = []
    missing: list[str] = []

    for kw in keywords:
        if kw in lower_resume or kw.replace(" ", "") in lower_resume.replace(" ", ""):
            matched.append(kw)
        else:
            missing.append(kw)

    score = (len(matched) / len(keywords)) * 100 if keywords else 0.0
    return round(score, 2), {
        "matched_keywords": matched,
        "missing_keywords": missing,
        "jd_keywords": keywords,
    }


def normalize_text_for_tfidf(text: str) -> str:
    """Light cleanup before TF-IDF vectorization."""
    cleaned = re.sub(r"\s+", " ", text)
    return cleaned.strip()
