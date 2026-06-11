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
    raw_readme: str = ""
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "project_name": self.project_name,
            "domain": self.domain,
            "frameworks": self.frameworks,
            "datasets": self.datasets,
            "metrics": self.metrics,
        }


def parse_github_url(url: str) -> tuple[str, str]:
    """Extract owner and repo from a GitHub URL."""
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
    """Fetch and decode repository README via GitHub REST API."""
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


def _extract_tech_from_readme(readme: str) -> list[str]:
    """Pull likely framework/tech names from README text."""
    known = {
        "python", "javascript", "typescript", "react", "vue", "angular", "django",
        "flask", "fastapi", "tensorflow", "pytorch", "pandas", "numpy", "docker",
        "kubernetes", "streamlit", "node.js", "nodejs", "spring", "rust", "go",
    }
    lower = readme.lower()
    found = [tech for tech in known if re.search(rf"\b{re.escape(tech)}\b", lower)]
    return found


def analyze_github_project(
    github_url: str,
    llm_client: Optional[LLMClient] = None,
) -> ProjectAnalysis:
    """
    Analyze a GitHub repository README and return structured project info.

    Uses GitHub REST API for README fetch; Gemini structures the output.
    Optional GITHUB_TOKEN raises rate limits from 60 to 5000 req/hr.
    """
    result = ProjectAnalysis()

    try:
        owner, repo = parse_github_url(github_url)
        result.project_name = repo
    except ValueError as exc:
        result.errors.append(str(exc))
        return result

    try:
        readme = fetch_readme(owner, repo)
        result.raw_readme = readme
    except Exception as exc:
        logger.error("Failed to fetch README: %s", exc)
        result.errors.append(f"Failed to fetch README: {exc}")
        return result

    tech_hints = _extract_tech_from_readme(readme)

    try:
        client = llm_client or get_llm_client()
        structured = _gemini_structure_project(client, repo, readme, tech_hints)
        result.project_name = structured.get("project_name", repo) or repo
        result.domain = structured.get("domain", MISSING_INFO)
        result.frameworks = structured.get("frameworks", tech_hints) or tech_hints
        result.datasets = structured.get("datasets", [])
        result.metrics = structured.get("metrics", [])

        # Enforce no fabrication for empty lists
        if not result.datasets:
            result.datasets = [MISSING_INFO]
        if not result.metrics:
            result.metrics = [MISSING_INFO]

    except Exception as exc:
        logger.error("Gemini structuring failed: %s", exc)
        result.errors.append(f"Analysis structuring failed: {exc}")
        result.frameworks = tech_hints or [MISSING_INFO]
        result.domain = MISSING_INFO
        result.datasets = [MISSING_INFO]
        result.metrics = [MISSING_INFO]

    return result


def _gemini_structure_project(
    client: LLMClient,
    repo_name: str,
    readme: str,
    tech_hints: list[str],
) -> dict[str, Any]:
    """Ask Gemini to structure README content into JSON fields."""
    system = """You analyze GitHub project READMEs. Return ONLY valid JSON with keys:
project_name, domain, frameworks (list), datasets (list), metrics (list).

Rules:
- Use only information present in the README.
- If datasets are not mentioned, return an empty list for datasets.
- If metrics are not mentioned, return an empty list for metrics.
- Never invent numbers, dataset names, or performance metrics.
- frameworks should list technologies explicitly mentioned."""

    prompt = f"""Repository: {repo_name}
Detected technologies: {', '.join(tech_hints) if tech_hints else 'none'}

README:
{readme[:8000]}

Return JSON only."""

    text = client.generate_text(prompt=prompt, system=system, max_output_tokens=1024)

    # Strip markdown code fences if present
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        logger.warning("Gemini returned non-JSON; using fallback parsing.")
        return {
            "project_name": repo_name,
            "domain": MISSING_INFO,
            "frameworks": tech_hints,
            "datasets": [],
            "metrics": [],
        }
