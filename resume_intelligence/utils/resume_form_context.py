"""Build human-readable resume snippets for form autofill."""

from __future__ import annotations

import re
from typing import Any

from models.resume import ResumeData

SECTION_EXTRACT = re.compile(
    r"(?is)(?:^|\n)\s*(experience|work experience|employment|projects|education|skills)\s*\n(.*?)(?=\n\s*(?:experience|work experience|employment|projects|education|skills|certifications)\s*\n|\Z)",
)


def format_resume_narrative(resume: ResumeData) -> str:
    """Plain-text resume summary for LLM prompts — uses structured fields + raw_text."""
    blocks: list[str] = []

    if resume.name:
        blocks.append(f"Name: {resume.name}")
    if resume.email:
        blocks.append(f"Email: {resume.email}")
    if resume.links:
        blocks.append("Links: " + ", ".join(resume.links))

    skills = synthesize_skills_answer(resume)
    if skills:
        blocks.append(f"Skills: {skills}")

    edu = synthesize_education_answer(resume)
    if edu:
        blocks.append(f"Education: {edu}")

    exp = synthesize_experience_answer(resume)
    if exp:
        blocks.append(f"Experience: {exp}")

    proj = synthesize_projects_answer(resume)
    if proj:
        blocks.append(f"Projects: {proj}")

    excerpt = resume.raw_text.strip()[:5000]
    if excerpt:
        blocks.append(f"Full resume text excerpt:\n{excerpt}")

    return "\n\n".join(blocks)


def synthesize_experience_answer(resume: ResumeData) -> str:
    parts: list[str] = []
    for entry in resume.experience:
        chunk = _format_entry(entry)
        if chunk:
            parts.append(chunk)

    if not parts:
        section = _extract_section(resume.raw_text, ("experience", "work experience", "employment"))
        if section:
            parts.append(_clean_section_text(section))

    return " ".join(parts)


def synthesize_projects_answer(resume: ResumeData) -> str:
    parts: list[str] = []
    for entry in resume.projects:
        chunk = _format_entry(entry)
        if chunk:
            parts.append(chunk)

    if not parts:
        section = _extract_section(resume.raw_text, ("projects", "personal projects"))
        if section:
            parts.append(_clean_section_text(section))

    return " ".join(parts)


def synthesize_education_answer(resume: ResumeData) -> str:
    parts: list[str] = []
    for entry in resume.education:
        chunk = _format_entry(entry)
        if chunk:
            parts.append(chunk)

    if not parts:
        section = _extract_section(resume.raw_text, ("education", "academic"))
        if section:
            parts.append(_clean_section_text(section))

    return " ".join(parts)


def synthesize_skills_answer(resume: ResumeData) -> str:
    if resume.skills:
        return ", ".join(resume.skills)

    section = _extract_section(resume.raw_text, ("skills", "technical skills", "technologies"))
    if section:
        return _clean_section_text(section).replace("\n", ", ")[:500]
    return ""


def _format_entry(entry: dict[str, Any]) -> str:
    title = str(entry.get("title", "") or "").strip()
    org = str(entry.get("org", "") or "").strip()
    dates = str(entry.get("dates", "") or "").strip()
    desc = str(entry.get("description", "") or "").strip()

    header = " — ".join(x for x in (title, org, dates) if x)
    if header and desc:
        return f"{header}: {desc}"
    if desc:
        return desc
    return header


def _extract_section(raw_text: str, headers: tuple[str, ...]) -> str:
    if not raw_text.strip():
        return ""
    lower = raw_text.lower()
    for header in headers:
        pattern = rf"(?is)(?:^|\n)\s*{re.escape(header)}\s*\n(.*?)(?=\n\s*(?:experience|work experience|employment|projects|education|skills|certifications|achievements)\b|\Z)"
        match = re.search(pattern, raw_text)
        if match:
            return match.group(1).strip()
        idx = lower.find(header)
        if idx >= 0:
            start = idx + len(header)
            return raw_text[start : start + 1200].strip()
    return ""


def _clean_section_text(text: str) -> str:
    lines = [re.sub(r"^[\u2022\-\*•]\s*", "", ln.strip()) for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]
    return " ".join(lines)[:1200]
