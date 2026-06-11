"""ATS scoring engine — 8-dimension resume analysis.

Dimensions (equal weight, 12.5% each):
1. Impact language
2. Quantification
3. ATS keyword alignment (TF-IDF)
4. Section completeness
5. Action verb density
6. Skill relevance
7. Formatting
8. Role-fit (TF-IDF document similarity)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from backend.matcher import analyze_skill_gap, extract_skills_from_text, normalize_skill
from backend.tfidf_scorer import keyword_alignment_score, normalize_text_for_tfidf, tfidf_document_similarity
from models.resume import ResumeData
from utils.logger import get_logger

logger = get_logger(__name__)

DIMENSION_WEIGHT = 1.0 / 8.0

IMPACT_WORDS = {
    "achieved", "improved", "increased", "reduced", "delivered", "led", "optimized",
    "streamlined", "accelerated", "enhanced", "saved", "grew", "scaled", "launched",
    "drove", "exceeded", "outperformed", "transformed",
}

ACTION_VERBS = {
    "achieved", "analyzed", "built", "created", "delivered", "designed", "developed",
    "engineered", "implemented", "improved", "increased", "led", "managed", "optimized",
    "reduced", "resolved", "spearheaded", "automated", "deployed", "architected",
}

QUANTIFICATION_PATTERNS = [
    re.compile(r"\d+%"),
    re.compile(r"\$\d[\d,]*"),
    re.compile(r"\b\d+\+?\s*(users|customers|clients|requests|transactions)\b", re.I),
    re.compile(r"\b\d+[\d,]*\s*(ms|seconds|minutes|hours|days)\b", re.I),
    re.compile(r"\b\d+x\b", re.I),
    re.compile(r"\b\d[\d,]*\+?\b"),
]

DIMENSION_NAMES = [
    "impact_language",
    "quantification",
    "ats_keyword_alignment",
    "section_completeness",
    "action_verb_density",
    "skill_relevance",
    "formatting",
    "role_fit",
]


@dataclass
class ATSScoreResult:
    """ATS scoring breakdown across 8 dimensions."""

    ats_score: float
    dimensions: dict[str, float] = field(default_factory=dict)
    breakdown: dict[str, Any] = field(default_factory=dict)

    # Convenience accessors for UI backward compatibility
    @property
    def skill_match_score(self) -> float:
        return self.dimensions.get("skill_relevance", 0.0)

    @property
    def semantic_similarity_score(self) -> float:
        return self.dimensions.get("role_fit", 0.0)

    @property
    def experience_relevance_score(self) -> float:
        return self.dimensions.get("ats_keyword_alignment", 0.0)


def compute_ats_score(resume: ResumeData, job_description: str) -> ATSScoreResult:
    """
    Compute ATS score on a 0-100 scale across 8 dimensions.

    Each dimension contributes 12.5% to the overall score.
    Keyword alignment and role-fit use TF-IDF similarity (scikit-learn).
    """
    if not job_description.strip():
        return ATSScoreResult(
            ats_score=0.0,
            dimensions={name: 0.0 for name in DIMENSION_NAMES},
            breakdown={"error": "Job description is empty."},
        )

    resume_text = normalize_text_for_tfidf(
        resume.raw_text.strip() or _fallback_resume_text(resume)
    )
    bullet_text = _collect_bullet_text(resume)

    gap = analyze_skill_gap(resume, job_description)
    kw_score, kw_breakdown = keyword_alignment_score(resume_text, job_description)
    role_fit_sim = tfidf_document_similarity(resume_text, job_description)

    dimensions = {
        "impact_language": _score_impact_language(bullet_text),
        "quantification": _score_quantification(bullet_text),
        "ats_keyword_alignment": kw_score,
        "section_completeness": _score_section_completeness(resume),
        "action_verb_density": _score_action_verbs(bullet_text),
        "skill_relevance": gap.match_percent,
        "formatting": _score_formatting(resume),
        "role_fit": round(role_fit_sim * 100, 2),
    }

    ats_score = round(sum(dimensions.values()) * DIMENSION_WEIGHT, 2)

    return ATSScoreResult(
        ats_score=ats_score,
        dimensions=dimensions,
        breakdown={
            "weights": {name: DIMENSION_WEIGHT for name in DIMENSION_NAMES},
            "dimension_labels": {
                "impact_language": "Impact Language",
                "quantification": "Quantification",
                "ats_keyword_alignment": "ATS Keyword Alignment",
                "section_completeness": "Section Completeness",
                "action_verb_density": "Action Verb Density",
                "skill_relevance": "Skill Relevance",
                "formatting": "Formatting",
                "role_fit": "Role-Fit",
            },
            "tfidf_keywords": kw_breakdown,
            "matched_skills": gap.matched_skills,
            "missing_skills": gap.missing_skills,
            "required_skills_count": len(extract_skills_from_text(job_description)),
            "resume_skills_count": len({normalize_skill(s) for s in resume.skills}),
        },
    )


def _collect_bullet_text(resume: ResumeData) -> str:
    parts: list[str] = []
    for entry in resume.experience + resume.projects:
        for key in ("description", "title"):
            value = entry.get(key, "")
            if value:
                parts.append(str(value))
    return " ".join(parts)


def _score_impact_language(text: str) -> float:
    """Score presence of impact-oriented language in bullets."""
    if not text.strip():
        return 0.0

    words = set(re.findall(r"[a-zA-Z]+", text.lower()))
    hits = len(words & IMPACT_WORDS)
    # 3+ impact words → strong score
    return round(min(100.0, hits * 25.0), 2)


def _score_quantification(text: str) -> float:
    """Score use of numbers and measurable outcomes."""
    if not text.strip():
        return 0.0

    hits = 0
    for pattern in QUANTIFICATION_PATTERNS:
        hits += len(pattern.findall(text))

    return round(min(100.0, hits * 20.0), 2)


def _score_section_completeness(resume: ResumeData) -> float:
    """Score whether key resume sections are present."""
    checks = [
        bool(resume.name.strip()),
        bool(resume.email.strip()),
        bool(resume.skills),
        bool(resume.experience),
        bool(resume.education),
        bool(resume.projects),
    ]
    return round((sum(checks) / len(checks)) * 100, 2)


def _score_action_verbs(text: str) -> float:
    """Score density of strong action verbs in experience/project text."""
    if not text.strip():
        return 0.0

    words = re.findall(r"[a-zA-Z]+", text.lower())
    if not words:
        return 0.0

    verb_hits = sum(1 for w in words if w in ACTION_VERBS)
    density = verb_hits / len(words)
    # ~5% action verb density → 100 score
    return round(min(100.0, density * 2000), 2)


def _score_formatting(resume: ResumeData) -> float:
    """Score basic formatting and contact completeness."""
    score = 0.0
    if resume.name.strip():
        score += 20
    if resume.email.strip():
        score += 20
    if len(resume.skills) >= 3:
        score += 20
    if len(resume.raw_text) >= 200:
        score += 20
    if resume.experience or resume.projects:
        score += 20
    return score


def _fallback_resume_text(resume: ResumeData) -> str:
    """Build text from structured fields when raw_text is empty."""
    parts = [resume.name, resume.email]
    parts.extend(resume.skills)
    for entry in resume.experience + resume.projects + resume.education:
        parts.extend(str(v) for v in entry.values() if v)
    return " ".join(parts)
