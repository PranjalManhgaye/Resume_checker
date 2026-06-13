"""GitHub repository project analyzer."""

from __future__ import annotations

import base64
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Optional
from urllib.parse import urlparse

import requests
from dotenv import load_dotenv

from services.llm_client import MISSING_INFO, LLMClient, get_llm_client
from utils.logger import get_logger

load_dotenv()
logger = get_logger(__name__)

GITHUB_API = "https://api.github.com"


@dataclass
class ProjectAnalysis:
    """Structured GitHub project analysis."""

    project_name: str = ""
    domain: str = ""
    frameworks: list[str] = field(default_factory=list)
    datasets: list[str] = field(default_factory=list)
    metrics: list[str] = field(default_factory=list)
    languages: dict[str, float] = field(default_factory=dict)
    stars: int = 0
    topics: list[str] = field(default_factory=list)
    last_updated: str = ""
    has_tests_hint: bool = False
    talking_points: list[str] = field(default_factory=list)
    interview_angle: str = ""
    raw_readme: str = ""
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "project_name": self.project_name,
            "domain": self.domain,
            "frameworks": self.frameworks,
            "datasets": self.datasets,
            "metrics": self.metrics,
            "languages": self.languages,
            "stars": self.stars,
            "topics": self.topics,
            "last_updated": self.last_updated,
            "has_tests_hint": self.has_tests_hint,
            "talking_points": self.talking_points,
            "interview_angle": self.interview_angle,
        }


def parse_github_url(url: str) -> tuple[str, str]:
    parsed = urlparse(url.strip())
    path_parts = [p for p in parsed.path.strip("/").split("/") if p]
    if len(path_parts) < 2:
        raise ValueError(f"Invalid GitHub URL: {url}")
    owner, repo = path_parts[0], path_parts[1]
    repo = repo.removesuffix(".git")
    return owner, repo


def _github_headers() -> dict[str, str]:
    headers = {"Accept": "application/vnd.github+json"}
    token = os.getenv("GITHUB_TOKEN", "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def fetch_readme(owner: str, repo: str) -> str:
    url = f"{GITHUB_API}/repos/{owner}/{repo}/readme"
    response = requests.get(url, headers=_github_headers(), timeout=30)
    if response.status_code == 404:
        raise ValueError(f"README not found for {owner}/{repo}")
    response.raise_for_status()
    data = response.json()
    content = data.get("content", "")
    encoding = data.get("encoding", "base64")
    if encoding == "base64":
        return base64.b64decode(content).decode("utf-8", errors="replace")
    return content


def fetch_repo_metadata(owner: str, repo: str) -> dict[str, Any]:
    """Fetch repo stats and language breakdown from GitHub API."""
    meta: dict[str, Any] = {
        "stars": 0,
        "topics": [],
        "last_updated": "",
        "languages": {},
        "description": "",
    }
    repo_url = f"{GITHUB_API}/repos/{owner}/{repo}"
    resp = requests.get(repo_url, headers=_github_headers(), timeout=30)
    if resp.ok:
        data = resp.json()
        meta["stars"] = data.get("stargazers_count", 0)
        meta["topics"] = data.get("topics", []) or []
        meta["last_updated"] = (data.get("updated_at") or "")[:10]
        meta["description"] = data.get("description") or ""

    lang_url = f"{GITHUB_API}/repos/{owner}/{repo}/languages"
    lang_resp = requests.get(lang_url, headers=_github_headers(), timeout=30)
    if lang_resp.ok:
        raw_langs = lang_resp.json()
        total = sum(raw_langs.values()) or 1
        meta["languages"] = {k: round(v / total * 100, 1) for k, v in raw_langs.items()}

    return meta


def _extract_tech_from_readme(readme: str) -> list[str]:
    known = {
        "python", "javascript", "typescript", "react", "vue", "angular", "django",
        "flask", "fastapi", "tensorflow", "pytorch", "pandas", "numpy", "docker",
        "kubernetes", "streamlit", "node.js", "nodejs", "spring", "rust", "go",
        "pytest", "scikit-learn", "postgresql", "mongodb",
    }
    lower = readme.lower()
    return [tech for tech in known if re.search(rf"\b{re.escape(tech)}\b", lower)]


def _detect_tests_hint(readme: str) -> bool:
    lower = readme.lower()
    hints = ("pytest", "unittest", "jest", "mocha", "ci badge", "github actions", "coverage")
    return any(h in lower for h in hints)


def analyze_github_project(
    github_url: str,
    llm_client: Optional[LLMClient] = None,
) -> ProjectAnalysis:
    result = ProjectAnalysis()

    try:
        owner, repo = parse_github_url(github_url)
        result.project_name = repo
    except ValueError as exc:
        result.errors.append(str(exc))
        return result

    metadata: dict[str, Any] = {}
    try:
        metadata = fetch_repo_metadata(owner, repo)
        result.stars = int(metadata.get("stars", 0))
        result.topics = list(metadata.get("topics", []))
        result.last_updated = str(metadata.get("last_updated", ""))
        result.languages = dict(metadata.get("languages", {}))
    except Exception as exc:
        logger.warning("Repo metadata fetch failed: %s", exc)
        result.errors.append(f"Could not fetch repo stats: {exc}")

    try:
        readme = fetch_readme(owner, repo)
        result.raw_readme = readme
    except Exception as exc:
        logger.error("Failed to fetch README: %s", exc)
        result.errors.append(f"Failed to fetch README: {exc}")
        return result

    tech_hints = _extract_tech_from_readme(readme)
    lang_names = list(result.languages.keys())
    merged_tech = list(dict.fromkeys(tech_hints + lang_names))
    result.has_tests_hint = _detect_tests_hint(readme)

    try:
        client = llm_client or get_llm_client()
        structured = _structure_project_analysis(
            client, repo, readme, merged_tech, metadata.get("description", "")
        )
        result.project_name = structured.get("project_name", repo) or repo
        result.domain = structured.get("domain", MISSING_INFO)
        result.frameworks = structured.get("frameworks", merged_tech) or merged_tech
        result.datasets = structured.get("datasets", []) or [MISSING_INFO]
        result.metrics = structured.get("metrics", []) or [MISSING_INFO]
        result.talking_points = structured.get("talking_points", [])
        result.interview_angle = structured.get("interview_angle", "")

        if not result.datasets or result.datasets == []:
            result.datasets = [MISSING_INFO]
        if not result.metrics or result.metrics == []:
            result.metrics = [MISSING_INFO]
        if not result.talking_points:
            result.talking_points = _fallback_talking_points(repo, merged_tech, result.has_tests_hint)

    except Exception as exc:
        logger.error("Project structuring failed: %s", exc)
        result.errors.append(f"Analysis structuring failed: {exc}")
        result.frameworks = merged_tech or [MISSING_INFO]
        result.domain = MISSING_INFO
        result.datasets = [MISSING_INFO]
        result.metrics = [MISSING_INFO]
        result.talking_points = _fallback_talking_points(repo, merged_tech, result.has_tests_hint)

    return result


def _structure_project_analysis(
    client: LLMClient,
    repo_name: str,
    readme: str,
    tech_hints: list[str],
    description: str,
) -> dict[str, Any]:
    system = """You analyze GitHub projects for interview prep. Return ONLY valid JSON with keys:
project_name, domain, frameworks (list), datasets (list), metrics (list),
talking_points (list of 3 short bullets), interview_angle (one sentence).

Rules:
- Use only README + provided metadata.
- Never invent dataset names, accuracy %, or user counts.
- talking_points: what a candidate could say in "tell me about a project"
- interview_angle: why this project matters for a software role"""

    prompt = f"""Repository: {repo_name}
Description: {description or 'none'}
Technologies detected: {', '.join(tech_hints) if tech_hints else 'none'}

README:
{readme[:7000]}

Return JSON only."""

    parsed = client.generate_json(prompt=prompt, system=system, max_output_tokens=1536)
    if isinstance(parsed, dict):
        return parsed
    raise ValueError("Expected JSON object from project analyzer")


def _fallback_talking_points(repo: str, tech: list[str], has_tests: bool) -> list[str]:
    points = [f"Built {repo} using {', '.join(tech[:4]) or 'open-source stack'}."]
    if has_tests:
        points.append("Includes automated tests or CI — mention reliability and quality.")
    points.append("Explain your specific contribution and one technical challenge you solved.")
    return points
