"""Shared data models for parsed resume content."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional


@dataclass
class ResumeData:
    """Structured resume representation consumed by all backend features."""

    name: str = ""
    email: str = ""
    education: list[dict[str, Any]] = field(default_factory=list)
    skills: list[str] = field(default_factory=list)
    projects: list[dict[str, Any]] = field(default_factory=list)
    experience: list[dict[str, Any]] = field(default_factory=list)
    links: list[str] = field(default_factory=list)
    raw_text: str = ""
    parse_warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to the JSON shape expected by the API contract."""
        return asdict(self)

    def to_llm_context(self) -> dict[str, Any]:
        """Slim payload for Gemini — omits raw_text to reduce latency."""
        return {
            "name": self.name,
            "email": self.email,
            "education": self.education,
            "skills": self.skills,
            "projects": self.projects,
            "experience": self.experience,
            "links": self.links,
        }


@dataclass
class ParseResult:
    """Outcome of parsing a resume file."""

    success: bool
    data: Optional[ResumeData] = None
    errors: list[str] = field(default_factory=list)
