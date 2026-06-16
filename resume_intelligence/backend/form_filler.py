"""Application form autofill from parsed resume data."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Optional

from models.resume import ResumeData
from services.llm_client import MISSING_INFO, LLMAPIError, LLMClient, get_llm_client
from utils.logger import get_logger
from utils.resume_form_context import (
    format_resume_narrative,
    synthesize_education_answer,
    synthesize_experience_answer,
    synthesize_projects_answer,
    synthesize_skills_answer,
)

logger = get_logger(__name__)

CGPA_PATTERN = re.compile(
    r"\b(?:CGPA|GPA|G\.P\.A\.?)\s*[:\-]?\s*(\d{1,2}(?:\.\d{1,2})?)\s*(?:/\s*\d{1,2})?\b",
    re.IGNORECASE,
)
YEAR_PATTERN = re.compile(r"\b(20\d{2}|19\d{2})\b")

YES_NO_PATTERN = re.compile(
    r"\b(are you|do you|have you|can you|will you|is this|yes/no)\b",
    re.IGNORECASE,
)

# Avoid matching "username", "program name", etc.
NAME_QUESTION = re.compile(
    r"\b(full name|your name|candidate name|legal name|name\s*\?)\b",
    re.IGNORECASE,
)


@dataclass
class FormQuestion:
    """A single application form question."""

    question: str
    field_type: Optional[str] = None


SYSTEM_PROMPT = """You are an application form assistant. Answer using ONLY the resume data provided.

Rules:
- Use facts from the resume text only.
- If the answer is not in the resume, respond exactly with: "Information not available."
- Never invent numbers, dates, achievements, or employers.
- Yes/no questions: answer "Yes" or "No" only when clearly supported; otherwise "Information not available."
- Short fields: one line max. Long text: 2-4 sentences max.
- Return valid JSON only: an object mapping each question string to its answer.
"""


def fill_form(
    resume: ResumeData,
    questions: list[FormQuestion],
    llm_client: Optional[LLMClient] = None,
    job_description: str = "",
) -> dict[str, str]:
    """
    Generate answers for application form questions.

    Uses resume synthesis first, then LLM for open-ended items.
    """
    if not questions:
        return {}

    answers: dict[str, str] = {}
    llm_needed: list[FormQuestion] = []

    for q in questions:
        deterministic = _try_deterministic_answer(resume, q, job_description)
        if deterministic is not None:
            answers[q.question] = deterministic
        else:
            llm_needed.append(q)

    if not llm_needed:
        return answers

    client = llm_client or get_llm_client()
    batch_answers = _llm_answer_batch(client, resume, llm_needed, job_description)
    answers.update(batch_answers)
    return answers


def _try_deterministic_answer(
    resume: ResumeData,
    question: FormQuestion,
    job_description: str = "",
) -> Optional[str]:
    q_lower = question.question.lower()

    if NAME_QUESTION.search(question.question) or q_lower.strip() in {"name?", "name"}:
        return resume.name or MISSING_INFO

    if any(k in q_lower for k in ("email", "e-mail")):
        return resume.email or MISSING_INFO

    if any(k in q_lower for k in ("cgpa", "gpa", "grade point")):
        cgpa = _extract_cgpa(resume)
        return cgpa if cgpa else MISSING_INFO

    if any(k in q_lower for k in ("linkedin",)) and "github" not in q_lower:
        for link in resume.links:
            if "linkedin" in link.lower():
                return link
        return MISSING_INFO

    if "github" in q_lower:
        for link in resume.links:
            if "github" in link.lower():
                return link
        return MISSING_INFO

    if any(k in q_lower for k in ("portfolio", "website", "personal site")):
        return resume.links[0] if resume.links else MISSING_INFO

    if any(k in q_lower for k in ("graduation", "graduate year", "year of passing", "year of graduation")):
        year = _extract_graduation_year(resume)
        return year if year else MISSING_INFO

    if any(k in q_lower for k in ("phone", "mobile", "contact number", "phone number")):
        phone = _extract_phone(resume.raw_text)
        return phone if phone else MISSING_INFO

    if any(k in q_lower for k in ("project title", "name of project")):
        if resume.projects:
            return str(resume.projects[0].get("title", MISSING_INFO))
        return MISSING_INFO

    if any(k in q_lower for k in ("project", "portfolio piece", "side project", "built a project")) and "title" not in q_lower:
        proj = synthesize_projects_answer(resume)
        return proj if proj else MISSING_INFO

    if any(k in q_lower for k in ("skill", "programming language", "tools you know")) or (
        "technolog" in q_lower and "project" not in q_lower
    ):
        skills = synthesize_skills_answer(resume)
        return skills if skills else MISSING_INFO

    if any(k in q_lower for k in ("education", "degree", "university", "college", "qualification")):
        edu = synthesize_education_answer(resume)
        return edu if edu else MISSING_INFO

    if any(
        k in q_lower
        for k in (
            "work experience",
            "professional experience",
            "employment history",
            "job history",
            "describe your experience",
            "previous experience",
            "internship",
        )
    ):
        exp = synthesize_experience_answer(resume)
        return exp if exp else MISSING_INFO

    if "why" in q_lower and job_description.strip():
        # Still needs LLM but only when we have JD context
        return None

    return None


def _extract_cgpa(resume: ResumeData) -> str:
    search_text = resume.raw_text
    for edu in resume.education:
        search_text += " " + " ".join(str(v) for v in edu.values())
    match = CGPA_PATTERN.search(search_text)
    return match.group(1) if match else ""


def _extract_graduation_year(resume: ResumeData) -> str:
    years: list[str] = []
    for edu in resume.education:
        for value in edu.values():
            for match in YEAR_PATTERN.findall(str(value)):
                years.append(match)
    if not years and resume.raw_text:
        for match in YEAR_PATTERN.findall(resume.raw_text):
            years.append(match)
    return max(years) if years else ""


def _extract_phone(text: str) -> str:
    pattern = re.compile(r"\+?\d{1,3}[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}")
    match = pattern.search(text)
    return match.group(0) if match else ""


def _resume_context(resume: ResumeData, job_description: str = "") -> dict[str, Any]:
    ctx: dict[str, Any] = {
        "structured": resume.to_llm_context(),
        "narrative": format_resume_narrative(resume),
    }
    if job_description.strip():
        ctx["target_job_description"] = job_description[:2000]
    return ctx


def _llm_answer_batch(
    client: LLMClient,
    resume: ResumeData,
    questions: list[FormQuestion],
    job_description: str = "",
) -> dict[str, str]:
    try:
        return _llm_answer_batch_once(client, resume, questions, job_description)
    except LLMAPIError as exc:
        logger.warning("Batch form fill failed (%s), trying per-question.", exc)
        return _llm_answer_one_by_one(client, resume, questions, job_description)


def _llm_answer_batch_once(
    client: LLMClient,
    resume: ResumeData,
    questions: list[FormQuestion],
    job_description: str = "",
) -> dict[str, str]:
    resume_block = format_resume_narrative(resume)
    question_list = json.dumps([q.question for q in questions], indent=2)
    jd_block = ""
    if job_description.strip():
        jd_block = f"\nTarget job description:\n{job_description[:2000]}\n"

    prompt = f"""Resume:
{resume_block}
{jd_block}
Questions (use exact strings as JSON keys):
{question_list}

Return a JSON object mapping each question string exactly to its answer."""

    parsed = client.generate_json(prompt=prompt, system=SYSTEM_PROMPT, max_output_tokens=2048)

    if not isinstance(parsed, dict):
        raise LLMAPIError("Form fill returned non-object JSON")

    flat = _flatten_llm_answer_object(parsed)
    return _normalize_answers(flat, questions)


def _llm_answer_one_by_one(
    client: LLMClient,
    resume: ResumeData,
    questions: list[FormQuestion],
    job_description: str = "",
) -> dict[str, str]:
    answers: dict[str, str] = {}
    resume_block = format_resume_narrative(resume)
    jd_block = f"\nTarget job:\n{job_description[:1500]}" if job_description.strip() else ""

    for q in questions:
        qtype = _classify_question(q.question)
        prompt = f"""Resume:
{resume_block}
{jd_block}

Question: {q.question}
Answer type: {qtype}.

Return JSON: {{"answer": "..."}}"""

        try:
            parsed = client.generate_json(
                prompt=prompt,
                system=SYSTEM_PROMPT,
                max_output_tokens=512,
            )
            if isinstance(parsed, dict):
                value = parsed.get("answer")
                if value is None:
                    value = _lookup_answer_for_question(q.question, _flatten_llm_answer_object(parsed))
                answers[q.question] = _trim_answer(str(value or MISSING_INFO), qtype)
            else:
                answers[q.question] = MISSING_INFO
        except LLMAPIError:
            answers[q.question] = MISSING_INFO

    return answers


def _flatten_llm_answer_object(parsed: dict[str, Any]) -> dict[str, Any]:
    """Handle {answers: [{question, answer}]} or nested shapes from LLM."""
    if "answers" in parsed and isinstance(parsed["answers"], list):
        flat: dict[str, Any] = {}
        for item in parsed["answers"]:
            if isinstance(item, dict):
                q = item.get("question") or item.get("q")
                a = item.get("answer") or item.get("a")
                if q:
                    flat[str(q)] = a
        return flat
    return parsed


def _lookup_answer_for_question(question: str, parsed: dict[str, Any]) -> Any:
    if question in parsed:
        return parsed[question]

    q_norm = _normalize_question_key(question)
    best_key = None
    best_score = 0.0
    for key, value in parsed.items():
        if key in ("answer", "answers"):
            continue
        score = SequenceMatcher(None, q_norm, _normalize_question_key(str(key))).ratio()
        if score > best_score:
            best_score = score
            best_key = key

    if best_key and best_score >= 0.55:
        return parsed[best_key]
    return MISSING_INFO


def _normalize_question_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _normalize_answers(parsed: dict[str, Any], questions: list[FormQuestion]) -> dict[str, str]:
    answers: dict[str, str] = {}
    for q in questions:
        qtype = _classify_question(q.question)
        raw = _lookup_answer_for_question(q.question, parsed)
        answers[q.question] = _trim_answer(str(raw), qtype)
    return answers


def _classify_question(question: str) -> str:
    q = question.lower()
    if any(k in q for k in ("describe", "explain", "why", "tell us", "summary")):
        return "long"
    if YES_NO_PATTERN.search(q) and len(q.split()) <= 8:
        return "yes_no"
    if any(k in q for k in ("years", "how many", "number of", "cgpa", "gpa", "phone")):
        return "short"
    return "short"


def _trim_answer(answer: str, qtype: str) -> str:
    answer = answer.strip()
    if not answer or answer.lower() in {"none", "n/a", "null"}:
        return MISSING_INFO
    if answer == MISSING_INFO:
        return MISSING_INFO
    if qtype == "yes_no":
        low = answer.lower()
        if low.startswith("yes"):
            return "Yes"
        if low.startswith("no"):
            return "No"
        return answer.split(".")[0][:80]
    if qtype == "short":
        return answer.split("\n")[0][:200]
    return answer[:600]
