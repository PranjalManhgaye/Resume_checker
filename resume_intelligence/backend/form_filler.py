"""Application form autofill from parsed resume data."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Optional

from models.resume import ResumeData
from services.llm_client import MISSING_INFO, LLMClient, get_llm_client
from utils.logger import get_logger

logger = get_logger(__name__)

CGPA_PATTERN = re.compile(
    r"\b(?:CGPA|GPA|G\.P\.A\.?)\s*[:\-]?\s*(\d{1,2}(?:\.\d{1,2})?)\s*(?:/\s*\d{1,2})?\b",
    re.IGNORECASE,
)
YEAR_PATTERN = re.compile(r"\b(20\d{2}|19\d{2})\b")


@dataclass
class FormQuestion:
    """A single application form question."""

    question: str
    field_type: Optional[str] = None


SYSTEM_PROMPT = """You are an application form assistant. Answer questions using ONLY the resume data provided.

Rules:
- Use facts from the resume JSON only.
- If the answer is not in the resume, respond exactly with: "Information not available."
- Never invent numbers, dates, achievements, or employers.
- Keep answers concise and professional.
- Return valid JSON only: an object mapping each question string to its answer.
"""


def fill_form(
    resume: ResumeData,
    questions: list[FormQuestion],
    llm_client: Optional[LLMClient] = None,
) -> dict[str, str]:
    """
    Generate answers for application form questions.

    Tries deterministic extraction first, then one batched Gemini call for the rest.
    """
    if not questions:
        return {}

    answers: dict[str, str] = {}
    gemini_needed: list[FormQuestion] = []

    for q in questions:
        deterministic = _try_deterministic_answer(resume, q)
        if deterministic is not None:
            answers[q.question] = deterministic
        else:
            gemini_needed.append(q)

    if gemini_needed:
        client = llm_client or get_llm_client()
        batch_answers = _gemini_answer_batch(client, resume, gemini_needed)
        answers.update(batch_answers)

    return answers


def _try_deterministic_answer(resume: ResumeData, question: FormQuestion) -> Optional[str]:
    """Return an answer without calling Gemini when possible."""
    q_lower = question.question.lower()

    if any(k in q_lower for k in ("name", "full name", "your name")):
        return resume.name or MISSING_INFO

    if any(k in q_lower for k in ("email", "e-mail")):
        return resume.email or MISSING_INFO

    if any(k in q_lower for k in ("cgpa", "gpa", "grade point")):
        cgpa = _extract_cgpa(resume)
        return cgpa if cgpa else MISSING_INFO

    if any(k in q_lower for k in ("linkedin", "github", "portfolio", "website", "url")):
        for link in resume.links:
            if "linkedin" in q_lower and "linkedin" in link.lower():
                return link
            if "github" in q_lower and "github" in link.lower():
                return link
        return resume.links[0] if resume.links else MISSING_INFO

    if any(k in q_lower for k in ("graduation", "graduate year", "year of passing")):
        year = _extract_graduation_year(resume)
        return year if year else MISSING_INFO

    if any(k in q_lower for k in ("phone", "mobile", "contact number")):
        phone = _extract_phone(resume.raw_text)
        return phone if phone else MISSING_INFO

    if "project title" in q_lower:
        if resume.projects:
            return str(resume.projects[0].get("title", MISSING_INFO))
        return MISSING_INFO

    return None


def _extract_cgpa(resume: ResumeData) -> str:
    """Search education descriptions and raw text for CGPA."""
    search_text = resume.raw_text
    for edu in resume.education:
        search_text += " " + " ".join(str(v) for v in edu.values())

    match = CGPA_PATTERN.search(search_text)
    return match.group(1) if match else ""


def _extract_graduation_year(resume: ResumeData) -> str:
    """Find the most recent year mentioned in education entries."""
    years: list[str] = []
    for edu in resume.education:
        for value in edu.values():
            for match in YEAR_PATTERN.findall(str(value)):
                years.append(match)
    return max(years) if years else ""


def _extract_phone(text: str) -> str:
    pattern = re.compile(r"\+?\d{1,3}[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}")
    match = pattern.search(text)
    return match.group(0) if match else ""


def _gemini_answer_batch(
    client: LLMClient,
    resume: ResumeData,
    questions: list[FormQuestion],
) -> dict[str, str]:
    """Answer all open-ended questions in a single Gemini call."""
    resume_json = json.dumps(resume.to_llm_context(), indent=2)
    question_list = json.dumps([q.question for q in questions], indent=2)

    prompt = f"""Resume data:
{resume_json}

Questions:
{question_list}

Return a JSON object mapping each question string to its answer."""

    try:
        response = client.generate_text(
            prompt=prompt,
            system=SYSTEM_PROMPT,
            max_output_tokens=1024,
        )
        cleaned = _strip_json_fences(response)
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return {str(k): str(v) for k, v in parsed.items()}
    except json.JSONDecodeError as exc:
        logger.error("Form fill JSON parse failed: %s", exc)
    except Exception as exc:
        logger.error("Form fill Gemini call failed: %s", exc)
        raise

    return {q.question: MISSING_INFO for q in questions}


def _strip_json_fences(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned
