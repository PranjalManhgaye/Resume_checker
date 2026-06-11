"""ATS scoring engine combining skill match, semantic similarity, and experience relevance."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from backend.matcher import analyze_skill_gap, extract_skills_from_text, normalize_skill
from models.embeddings import get_embedding_model
from models.resume import ResumeData
from utils.logger import get_logger

logger = get_logger(__name__)

WEIGHT_SKILL = 0.40
WEIGHT_SEMANTIC = 0.40
WEIGHT_EXPERIENCE = 0.20


@dataclass
class ATSScoreResult:
    """ATS scoring breakdown."""

    ats_score: float
    skill_match_score: float
    semantic_similarity_score: float
    experience_relevance_score: float
    breakdown: dict[str, Any] = field(default_factory=dict)


def _experience_text(resume: ResumeData) -> str:
    """Concatenate experience descriptions for embedding comparison."""
    parts: list[str] = []
    for entry in resume.experience:
        for key in ("title", "org", "description"):
            value = entry.get(key, "")
            if value:
                parts.append(str(value))
    return " ".join(parts)


def compute_ats_score(resume: ResumeData, job_description: str) -> ATSScoreResult:
    """
    Compute ATS score on a 0-100 scale.

    Formula:
        40% skill match + 40% semantic similarity + 20% experience relevance

    Skill match uses overlap between resume skills and JD-extracted skills.
    Semantic similarity uses Sentence Transformers cosine similarity.
    Experience relevance compares experience text to the job description.
    """
    if not job_description.strip():
        return ATSScoreResult(
            ats_score=0.0,
            skill_match_score=0.0,
            semantic_similarity_score=0.0,
            experience_relevance_score=0.0,
            breakdown={"error": "Job description is empty."},
        )

    gap = analyze_skill_gap(resume, job_description)
    skill_match = gap.match_percent / 100.0

    embedder = get_embedding_model()
    resume_text = resume.raw_text.strip() or _fallback_resume_text(resume)
    experience_text = _experience_text(resume)

    # Batch encode resume + JD pairs in one pass for faster scoring
    semantic_sim, experience_sim = embedder.batch_cosine_similarities(
        [resume_text, experience_text],
        [job_description, job_description],
    )

    weighted = (
        WEIGHT_SKILL * skill_match
        + WEIGHT_SEMANTIC * semantic_sim
        + WEIGHT_EXPERIENCE * experience_sim
    )
    ats_score = round(weighted * 100, 2)

    required_skills = extract_skills_from_text(job_description)
    resume_skills_norm = {normalize_skill(s) for s in resume.skills}

    return ATSScoreResult(
        ats_score=ats_score,
        skill_match_score=round(skill_match * 100, 2),
        semantic_similarity_score=round(semantic_sim * 100, 2),
        experience_relevance_score=round(experience_sim * 100, 2),
        breakdown={
            "weights": {
                "skill_match": WEIGHT_SKILL,
                "semantic_similarity": WEIGHT_SEMANTIC,
                "experience_relevance": WEIGHT_EXPERIENCE,
            },
            "matched_skills": gap.matched_skills,
            "missing_skills": gap.missing_skills,
            "required_skills_count": len(required_skills),
            "resume_skills_count": len(resume_skills_norm),
        },
    )


def _fallback_resume_text(resume: ResumeData) -> str:
    """Build text from structured fields when raw_text is empty."""
    parts = [resume.name, resume.email]
    parts.extend(resume.skills)
    for entry in resume.experience + resume.projects + resume.education:
        parts.extend(str(v) for v in entry.values() if v)
    return " ".join(parts)
