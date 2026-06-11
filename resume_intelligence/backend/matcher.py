"""Skill gap analysis between resume and job description."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from models.resume import ResumeData

# Common skill aliases for normalization
SKILL_ALIASES: dict[str, str] = {
    "js": "javascript",
    "ts": "typescript",
    "py": "python",
    "ml": "machine learning",
    "ai": "artificial intelligence",
    "nlp": "natural language processing",
    "k8s": "kubernetes",
    "postgres": "postgresql",
    "mongo": "mongodb",
    "node": "nodejs",
    "react.js": "react",
    "vue.js": "vue",
    "c++": "cpp",
    "c#": "csharp",
}

# Known tech terms to extract from job descriptions
KNOWN_SKILLS = {
    "python", "java", "javascript", "typescript", "react", "angular", "vue",
    "nodejs", "django", "flask", "fastapi", "spring", "sql", "postgresql",
    "mysql", "mongodb", "redis", "docker", "kubernetes", "aws", "azure", "gcp",
    "git", "ci/cd", "linux", "pandas", "numpy", "scikit-learn", "tensorflow",
    "pytorch", "machine learning", "deep learning", "nlp", "html", "css",
    "rest", "api", "graphql", "microservices", "agile", "scrum", "c++", "c",
    "go", "golang", "rust", "swift", "kotlin", "spark", "hadoop", "tableau",
    "power bi", "excel", "communication", "leadership", "problem solving",
    "data structures", "algorithms", "oop", "testing", "pytest", "junit",
    "selenium", "streamlit", "gemini", "llm", "transformers", "embeddings",
}


@dataclass
class SkillGapResult:
    """Skill comparison outcome."""

    matched_skills: list[str] = field(default_factory=list)
    missing_skills: list[str] = field(default_factory=list)
    recommended_skills: list[str] = field(default_factory=list)
    match_percent: float = 0.0


def normalize_skill(skill: str) -> str:
    """Normalize a skill string for comparison."""
    cleaned = skill.lower().strip()
    cleaned = re.sub(r"[^\w\s/+#.-]", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return SKILL_ALIASES.get(cleaned, cleaned)


def extract_skills_from_text(text: str) -> list[str]:
    """
    Extract likely skill keywords from free text (job description).

    Only returns skills found in the text — nothing invented.
    """
    if not text.strip():
        return []

    lower = text.lower()
    found: list[str] = []

    for skill in sorted(KNOWN_SKILLS, key=len, reverse=True):
        pattern = r"\b" + re.escape(skill) + r"\b"
        if re.search(pattern, lower):
            found.append(skill)

    return found


def analyze_skill_gap(resume: ResumeData, job_description: str) -> SkillGapResult:
    """Compare resume skills against skills required in a job description."""
    if not job_description.strip():
        return SkillGapResult()

    resume_skills = {normalize_skill(s) for s in resume.skills if s.strip()}
    required_skills = extract_skills_from_text(job_description)

    if not required_skills:
        return SkillGapResult()

    required_normalized = [normalize_skill(s) for s in required_skills]
    required_set = set(required_normalized)

    matched = sorted(resume_skills & required_set)
    missing = sorted(required_set - resume_skills)

    # Recommended = JD skills not on resume, ordered by frequency in JD
    freq: dict[str, int] = {}
    lower_jd = job_description.lower()
    for skill in required_normalized:
        freq[skill] = lower_jd.count(skill)

    recommended = sorted(missing, key=lambda s: freq.get(s, 0), reverse=True)

    match_percent = (len(matched) / len(required_set) * 100) if required_set else 0.0

    return SkillGapResult(
        matched_skills=matched,
        missing_skills=missing,
        recommended_skills=recommended,
        match_percent=round(match_percent, 2),
    )
