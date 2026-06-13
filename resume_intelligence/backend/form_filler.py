"""Application form autofill from parsed resume data."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Optional

from models.resume import ResumeData
from services.llm_client import MISSING_INFO, LLMAPIError, LLMClient, get_llm_client
from utils.logger import get_logger

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


@dataclass
class FormQuestion:
    """A single application form question."""

    question: str
    field_type: Optional[str] = None


SYSTEM_PROMPT = """You are an application form assistant. Answer using ONLY the resume data provided.

Rules:
- Use facts from the resume JSON only.
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
) -> dict[str, str]:
    """
    Generate answers for application form questions.

    Tries deterministic extraction first, then batched JSON LLM call,
    then per-question fallback if batch parsing fails.
    """
    if not questions:
        return {}

    answers: dict[str, str] = {}
    llm_needed: list[FormQuestion] = []

    for q in questions:
        deterministic = _try_deterministic_answer(resume, q)
        if deterministic is not None:
            answers[q.question] = deterministic
        else:
            llm_needed.append(q)

    if not llm_needed:
        return answers

    client = llm_client or get_llm_client()
    batch_answers = _llm_answer_batch(client, resume, llm_needed)
    answers.update(batch_answers)
    return answers


def _try_deterministic_answer(resume: ResumeData, question: FormQuestion) -> Optional[str]:
    """Return an answer without calling the LLM when possible."""
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
    return max(years) if years else ""


def _extract_phone(text: str) -> str:
    pattern = re.compile(r"\+?\d{1,3}[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}")
    match = pattern.search(text)
    return match.group(0) if match else ""


def _resume_context(resume: ResumeData) -> dict:
    """Richer context for open-ended questions — includes raw_text snippet."""
    ctx = resume.to_llm_context()
    if resume.raw_text.strip():
        ctx["raw_text_excerpt"] = resume.raw_text[:4000]
    return ctx


def _llm_answer_batch(
    client: LLMClient,
    resume: ResumeData,
    questions: list[FormQuestion],
) -> dict[str, str]:
    """Answer questions in one JSON call; fall back to per-question on failure."""
    try:
        return _llm_answer_batch_once(client, resume, questions)
    except LLMAPIError as exc:
        logger.warning("Batch form fill failed (%s), trying per-question.", exc)
        return _llm_answer_one_by_one(client, resume, questions)


def _llm_answer_batch_once(
    client: LLMClient,
    resume: ResumeData,
    questions: list[FormQuestion],
) -> dict[str, str]:
    resume_json = json.dumps(_resume_context(resume), indent=2)
    question_list = json.dumps([q.question for q in questions], indent=2)

    prompt = f"""Resume data:
{resume_json}

Questions:
{question_list}

Return a JSON object mapping each question string exactly to its answer."""

    parsed = client.generate_json(prompt=prompt, system=SYSTEM_PROMPT, max_output_tokens=1536)

    if not isinstance(parsed, dict):
        raise LLMAPIError("Form fill returned non-object JSON")

    return _normalize_answers(parsed, questions)


def _llm_answer_one_by_one(
    client: LLMClient,
    resume: ResumeData,
    questions: list[FormQuestion],
) -> dict[str, str]:
    answers: dict[str, str] = {}
    resume_json = json.dumps(_resume_context(resume), indent=2)

    for q in questions:
        qtype = _classify_question(q.question)
        type_hint = f"Answer type: {qtype}."
        prompt = f"""Resume data:
{resume_json}

Question: {q.question}
{type_hint}

Return JSON: {{"answer": "..."}}"""

        try:
            parsed = client.generate_json(
                prompt=prompt,
                system=SYSTEM_PROMPT,
                max_output_tokens=384,
            )
            if isinstance(parsed, dict) and "answer" in parsed:
                answers[q.question] = _trim_answer(str(parsed["answer"]), qtype)
            elif isinstance(parsed, dict) and q.question in parsed:
                answers[q.question] = _trim_answer(str(parsed[q.question]), qtype)
            else:
                answers[q.question] = MISSING_INFO
        except LLMAPIError:
            answers[q.question] = MISSING_INFO

    return answers


def _normalize_answers(parsed: dict, questions: list[FormQuestion]) -> dict[str, str]:
    answers: dict[str, str] = {}
    for q in questions:
        qtype = _classify_question(q.question)
        raw = parsed.get(q.question, MISSING_INFO)
        answers[q.question] = _trim_answer(str(raw), qtype)
    return answers


def _classify_question(question: str) -> str:
    q = question.lower()
    if YES_NO_PATTERN.search(q):
        return "yes_no"
    if any(k in q for k in ("years", "how many", "number of", "cgpa", "gpa", "phone")):
        return "short"
    if any(k in q for k in ("describe", "explain", "why", "tell us", "summary")):
        return "long"
    return "short"


def _trim_answer(answer: str, qtype: str) -> str:
    answer = answer.strip()
    if not answer:
        return MISSING_INFO
    if qtype == "yes_no":
        low = answer.lower()
        if low.startswith("yes"):
            return "Yes"
        if low.startswith("no"):
            return "No"
        if answer == MISSING_INFO:
            return MISSING_INFO
        return answer.split(".")[0][:80]
    if qtype == "short":
        return answer.split("\n")[0][:200]
    return answer[:600]
