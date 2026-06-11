"""Recruiter-style candidate summary generation."""

from __future__ import annotations

import json
from typing import Optional

from models.resume import ResumeData
from services.llm_client import MISSING_INFO, LLMClient, get_llm_client
from utils.logger import get_logger

logger = get_logger(__name__)

SYSTEM_PROMPT = """You write concise recruiter-style candidate summaries.

Rules:
- 2-3 sentences maximum.
- Use only facts from the provided resume data.
- Highlight skills, experience, and notable projects.
- Professional tone, third person.
- Never invent employers, achievements, or credentials.
- If insufficient data, respond with: "Information not available."
"""


def generate_candidate_summary(
    resume: ResumeData,
    job_description: str = "",
    llm_client: Optional[LLMClient] = None,
) -> str:
    """
    Generate a recruiter-style candidate summary.

    Falls back to a template summary if the Gemini API call fails.
    """
    if not _has_summary_data(resume):
        return MISSING_INFO

    try:
        client = llm_client or get_llm_client()
        resume_json = json.dumps(resume.to_llm_context(), indent=2)

        prompt = f"""Resume data:
{resume_json}
"""
        if job_description.strip():
            prompt += f"\nTarget job description:\n{job_description[:2000]}\n"

        prompt += "\nWrite a 2-3 sentence recruiter-style summary."

        return client.generate_text(
            prompt=prompt,
            system=SYSTEM_PROMPT,
            max_output_tokens=256,
        )

    except Exception as exc:
        logger.error("Candidate summary Gemini call failed: %s", exc)
        return _template_summary(resume)


def _has_summary_data(resume: ResumeData) -> bool:
    return bool(
        resume.name
        or resume.skills
        or resume.experience
        or resume.projects
        or resume.education
    )


def _template_summary(resume: ResumeData) -> str:
    """Simple fallback summary built only from parsed facts."""
    parts: list[str] = []

    if resume.name:
        parts.append(resume.name)
    else:
        parts.append("Candidate")

    if resume.experience:
        titles = [e.get("title", "") for e in resume.experience if e.get("title")]
        if titles:
            parts.append(f"with experience as {titles[0]}")

    if resume.skills:
        top_skills = ", ".join(resume.skills[:5])
        parts.append(f"skilled in {top_skills}")

    if resume.education:
        edu = resume.education[0]
        degree = edu.get("title", "")
        school = edu.get("org", "")
        if degree and school:
            parts.append(f"and education in {degree} from {school}")

    if len(parts) <= 1:
        return MISSING_INFO

    return " ".join(parts) + "."
