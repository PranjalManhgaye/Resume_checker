"""Pytest configuration and shared fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure project root is on the import path
ROOT = Path(__file__).resolve().parent.parent
TESTS_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

from llm_fixtures import FixtureLLMClient, RewriterFixtureLLMClient, load_llm_fixture


@pytest.fixture
def sample_resume_alex() -> "ResumeData":
    """Structured resume matching data/samples/sample_resume_well_formatted.txt."""
    from models.resume import ResumeData

    return ResumeData(
        name="Alex Johnson",
        email="alex.johnson@email.com",
        links=[
            "https://github.com/alexjohnson",
            "https://linkedin.com/in/alexjohnson",
        ],
        education=[
            {
                "title": "B.Tech Computer Science",
                "org": "State University",
                "dates": "2019 - 2023",
                "description": "CGPA: 8.33/10",
            }
        ],
        skills=[
            "Python",
            "Java",
            "SQL",
            "React",
            "Docker",
            "Git",
            "Machine Learning",
            "FastAPI",
        ],
        experience=[
            {
                "title": "Software Engineering Intern",
                "org": "TechCorp Inc.",
                "dates": "June 2022 - August 2022",
                "description": (
                    "Built REST APIs using FastAPI and PostgreSQL\n"
                    "Deployed services with Docker on AWS\n"
                    "Reduced API response time through query optimization"
                ),
            }
        ],
        projects=[
            {
                "title": "Resume Parser Tool",
                "org": "",
                "dates": "",
                "description": (
                    "Developed a PDF resume parser using Python and PyMuPDF\n"
                    "Implemented skill extraction with regex and section detection"
                ),
            },
            {
                "title": "E-Commerce Dashboard",
                "org": "",
                "dates": "",
                "description": (
                    "Built a React dashboard with data visualization\n"
                    "Integrated with REST backend APIs"
                ),
            },
        ],
    )


@pytest.fixture
def form_fill_mock_client() -> FixtureLLMClient:
    """Mock LLM client returning golden form-fill batch JSON."""
    return FixtureLLMClient(load_llm_fixture("form_fill_batch_response.json"))


@pytest.fixture
def rewriter_mock_client() -> RewriterFixtureLLMClient:
    """Mock LLM client returning golden rewriter JSON per bullet."""
    return RewriterFixtureLLMClient(load_llm_fixture("rewriter_response.json"))


@pytest.fixture
def summary_mock_client() -> FixtureLLMClient:
    """Mock LLM client returning golden summary text."""
    return FixtureLLMClient(load_llm_fixture("summary_response.txt"))


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "integration: live LLM API tests (skipped in CI; require API keys)",
    )
