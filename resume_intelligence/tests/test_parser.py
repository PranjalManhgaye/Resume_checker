"""Tests for resume parser."""

from __future__ import annotations

from pathlib import Path

import pytest

from backend.parser import parse_resume, parse_resume_from_path
from utils.file_utils import validate_resume_file

SAMPLES_DIR = Path(__file__).resolve().parent.parent / "data" / "samples"


@pytest.fixture(scope="module", autouse=True)
def ensure_samples() -> None:
    """Generate sample files if they don't exist."""
    docx_path = SAMPLES_DIR / "sample_resume_well_formatted.docx"
    if not docx_path.exists():
        from data.samples.generate_samples import main
        main()


def test_validate_resume_file_rejects_unsupported_type() -> None:
    result = validate_resume_file("resume.txt", b"hello")
    assert not result.valid
    assert "Unsupported" in (result.error or "")


def test_validate_resume_file_rejects_empty() -> None:
    result = validate_resume_file("resume.pdf", b"")
    assert not result.valid


def test_parse_well_formatted_docx_extracts_email() -> None:
    path = SAMPLES_DIR / "sample_resume_well_formatted.docx"
    result = parse_resume_from_path(str(path))

    assert result.success
    assert result.data is not None
    assert result.data.email == "alex.johnson@email.com"


def test_parse_well_formatted_docx_extracts_skills() -> None:
    path = SAMPLES_DIR / "sample_resume_well_formatted.docx"
    result = parse_resume_from_path(str(path))

    assert result.data is not None
    assert len(result.data.skills) > 0
    assert any("python" in s.lower() for s in result.data.skills)


def test_parse_well_formatted_docx_extracts_name() -> None:
    path = SAMPLES_DIR / "sample_resume_well_formatted.docx"
    result = parse_resume_from_path(str(path))

    assert result.data is not None
    assert "Alex" in result.data.name or "Johnson" in result.data.name


def test_parse_well_formatted_pdf() -> None:
    path = SAMPLES_DIR / "sample_resume_well_formatted.pdf"
    if not path.exists():
        pytest.skip("PDF sample not generated")

    result = parse_resume_from_path(str(path))
    assert result.data is not None
    assert result.data.email == "alex.johnson@email.com"


def test_parse_poorly_formatted_partial_success() -> None:
    path = SAMPLES_DIR / "sample_resume_poorly_formatted.docx"
    result = parse_resume_from_path(str(path))

    assert result.data is not None
    assert result.data.email == "jane.doe@example.com"


def test_parse_corrupt_file_graceful_failure() -> None:
    result = parse_resume(b"not a real pdf content", "broken.pdf")
    assert result.data is not None or result.errors
