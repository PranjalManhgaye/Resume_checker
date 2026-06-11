"""Resume improvement engine — rephrase bullets without fabricating facts."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from models.resume import ResumeData
from services.llm_client import MISSING_INFO, LLMAPIError, LLMClient, get_llm_client
from utils.logger import get_logger

logger = get_logger(__name__)

SYSTEM_PROMPT = """You are a professional resume writing coach.

Rules:
- Improve clarity, impact, and action-oriented language.
- Preserve truthfulness — never invent numbers, metrics, or achievements.
- Only rephrase facts already present in the original text.
- If a bullet lacks enough detail to improve, return it unchanged.
- Use strong action verbs where appropriate.
- Return valid JSON only with keys "experience" and "projects".
- Each key maps to an array of {"original": "...", "improved": "..."} objects."""


@dataclass
class ImprovedBullet:
    """Original and improved bullet pair."""

    section: str
    original: str
    improved: str
    unchanged: bool = False


@dataclass
class RewriteResult:
    """Collection of improved resume bullets."""

    experience: list[ImprovedBullet] = field(default_factory=list)
    projects: list[ImprovedBullet] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def improve_resume(
    resume: ResumeData,
    llm_client: Optional[LLMClient] = None,
) -> RewriteResult:
    """Generate improved experience and project bullet points in one API call."""
    result = RewriteResult()

    exp_bullets = _collect_bullets(resume.experience)
    proj_bullets = _collect_bullets(resume.projects)

    if not exp_bullets and not proj_bullets:
        result.errors.append(
            "No experience or project bullets found to improve. "
            "Ensure your resume has description bullets under each role."
        )
        return result

    try:
        client = llm_client or get_llm_client()
        parsed = _improve_all_bullets(client, exp_bullets, proj_bullets)
        result.experience = parsed.get("experience", [])
        result.projects = parsed.get("projects", [])
    except LLMAPIError as exc:
        result.errors.append(exc.user_message)
        result.experience = _fallback_bullets("experience", exp_bullets)
        result.projects = _fallback_bullets("projects", proj_bullets)
    except Exception as exc:
        logger.error("Resume rewrite failed: %s", exc)
        result.errors.append(str(exc))
        result.experience = _fallback_bullets("experience", exp_bullets)
        result.projects = _fallback_bullets("projects", proj_bullets)

    return result


def _collect_bullets(entries: list[dict[str, Any]]) -> list[str]:
    """Gather bullets from structured entries, combining title and org when needed."""
    bullets: list[str] = []
    for entry in entries:
        desc = entry.get("description", "").strip()
        title = entry.get("title", "").strip()
        org = entry.get("org", "").strip()
        dates = entry.get("dates", "").strip()

        if desc:
            bullets.append(desc)
        elif title:
            parts = [title]
            if org and org.lower() not in title.lower():
                parts.append(f"at {org}")
            if dates:
                parts.append(f"({dates})")
            bullets.append(" — ".join(parts))
    return bullets


def _improve_all_bullets(
    client: LLMClient,
    exp_bullets: list[str],
    proj_bullets: list[str],
) -> dict[str, list[ImprovedBullet]]:
    """Send experience and project bullets in one Gemini request."""
    prompt = f"""Improve these resume bullets.

Experience bullets:
{json.dumps(exp_bullets, indent=2)}

Project bullets:
{json.dumps(proj_bullets, indent=2)}

Return JSON with "experience" and "projects" arrays.
Each item must have "original" and "improved" keys."""

    response = client.generate_text(
        prompt=prompt,
        system=SYSTEM_PROMPT,
        max_output_tokens=2048,
    )
    cleaned = _strip_json_fences(response)
    parsed = json.loads(cleaned)

    return {
        "experience": _parse_bullet_list(parsed.get("experience", []), "experience", exp_bullets),
        "projects": _parse_bullet_list(parsed.get("projects", []), "projects", proj_bullets),
    }


def _parse_bullet_list(
    items: list[dict[str, Any]],
    section: str,
    originals: list[str],
) -> list[ImprovedBullet]:
    """Parse Gemini response into ImprovedBullet list."""
    if not items and originals:
        return _fallback_bullets(section, originals)

    bullets: list[ImprovedBullet] = []
    for item in items:
        original = item.get("original", "")
        improved = item.get("improved", MISSING_INFO)
        bullets.append(
            ImprovedBullet(
                section=section,
                original=original,
                improved=improved,
                unchanged=original.strip() == improved.strip(),
            )
        )
    return bullets


def _fallback_bullets(section: str, bullets: list[str]) -> list[ImprovedBullet]:
    """Return originals unchanged when AI improvement is unavailable."""
    return [
        ImprovedBullet(section=section, original=b, improved=b, unchanged=True)
        for b in bullets
    ]


def _strip_json_fences(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned
