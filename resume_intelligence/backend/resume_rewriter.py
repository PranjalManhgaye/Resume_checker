"""Resume improvement engine — rephrase bullets without fabricating facts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from models.resume import ResumeData
from services.llm_client import MISSING_INFO, LLMAPIError, LLMClient, get_llm_client
from utils.logger import get_logger

logger = get_logger(__name__)

BULLET_SYSTEM = """You are a professional resume coach (EnhanceCV-style clarity).

Rules:
- Rewrite for impact: strong past-tense action verb first, clear outcome when stated in original.
- Preserve every fact — never invent employers, tools, numbers, or metrics.
- If the bullet is only a project title with no actions, expand structure but do not add fake results.
- Keep similar length (1-2 lines).
- Return JSON only: {"original": "...", "improved": "..."}"""


@dataclass
class ImprovedBullet:
    """Original and improved bullet pair."""

    section: str
    original: str
    improved: str
    unchanged: bool = False
    score_hint: str = ""


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
    """Improve experience and project bullets one at a time for reliable JSON."""
    result = RewriteResult()

    exp_bullets = _collect_bullets(resume.experience)
    proj_bullets = _collect_bullets(resume.projects)

    if not exp_bullets and not proj_bullets:
        result.errors.append(
            "No experience or project bullets found to improve. "
            "Upload a resume with description text under each role or project."
        )
        return result

    try:
        client = llm_client or get_llm_client()
        result.experience = _improve_section(client, "experience", exp_bullets)
        result.projects = _improve_section(client, "projects", proj_bullets)
    except LLMAPIError as exc:
        result.errors.append(exc.user_message)
        result.experience = _fallback_bullets("experience", exp_bullets)
        result.projects = _fallback_bullets("projects", proj_bullets)

    changed = sum(1 for b in result.experience + result.projects if not b.unchanged)
    if not changed and not result.errors:
        result.errors.append(
            "AI could not improve these bullets — try adding more detail to descriptions "
            "or fix parser output on the Upload page."
        )

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
            for line in desc.split("\n"):
                line = line.strip().lstrip("•-* ").strip()
                if line:
                    bullets.append(line)
        elif title:
            parts = [title]
            if org and org.lower() not in title.lower():
                parts.append(f"at {org}")
            if dates:
                parts.append(f"({dates})")
            bullets.append(" — ".join(parts))
    return bullets


def _improve_section(
    client: LLMClient,
    section: str,
    bullets: list[str],
) -> list[ImprovedBullet]:
    improved: list[ImprovedBullet] = []
    for bullet in bullets:
        try:
            improved.append(_improve_single_bullet(client, section, bullet))
        except LLMAPIError as exc:
            logger.warning("Bullet rewrite failed: %s", exc)
            improved.append(
                ImprovedBullet(
                    section=section,
                    original=bullet,
                    improved=bullet,
                    unchanged=True,
                    score_hint="rewrite failed",
                )
            )
    return improved


def _improve_single_bullet(client: LLMClient, section: str, bullet: str) -> ImprovedBullet:
    prompt = f"""Section: {section}

Original bullet:
{bullet}

Rewrite for clarity and impact. Return JSON with "original" and "improved" keys."""

    parsed = client.generate_json(prompt=prompt, system=BULLET_SYSTEM, max_output_tokens=512)

    if not isinstance(parsed, dict):
        raise LLMAPIError("Invalid bullet response shape")

    original = str(parsed.get("original", bullet)).strip() or bullet
    improved = str(parsed.get("improved", bullet)).strip() or bullet

    if improved == MISSING_INFO:
        improved = bullet

    unchanged = _normalize(original) == _normalize(improved)
    return ImprovedBullet(
        section=section,
        original=bullet,
        improved=improved,
        unchanged=unchanged,
        score_hint="" if unchanged else "improved",
    )


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def _fallback_bullets(section: str, bullets: list[str]) -> list[ImprovedBullet]:
    """Return originals unchanged when AI improvement is unavailable."""
    return [
        ImprovedBullet(section=section, original=b, improved=b, unchanged=True)
        for b in bullets
    ]
