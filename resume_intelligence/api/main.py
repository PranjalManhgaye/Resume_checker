"""FastAPI REST layer for Resume Intelligence Platform."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.ats import compute_ats_score
from backend.candidate_summary import generate_candidate_summary
from backend.form_filler import FormQuestion, fill_form
from backend.matcher import analyze_skill_gap
from backend.parser import parse_resume
from backend.project_analyzer import analyze_github_project
from backend.resume_rewriter import improve_resume
from models.resume import ResumeData
from services.llm_client import LLMAPIError

app = FastAPI(
    title="Resume Intelligence API",
    description="REST API for resume parsing, ATS scoring, and AI features.",
    version="1.0.0",
)


class JobDescriptionRequest(BaseModel):
    resume: dict[str, Any]
    job_description: str


class FormFillRequest(BaseModel):
    resume: dict[str, Any]
    questions: list[str]
    job_description: str = ""


class SummaryRequest(BaseModel):
    resume: dict[str, Any]
    job_description: str = ""


class ProjectAnalyzeRequest(BaseModel):
    github_url: str


class RewriteRequest(BaseModel):
    resume: dict[str, Any]


def _resume_from_dict(data: dict[str, Any]) -> ResumeData:
    return ResumeData(**{k: v for k, v in data.items() if k in ResumeData.__dataclass_fields__})


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/parse")
async def parse_endpoint(file: UploadFile = File(...)) -> dict[str, Any]:
    file_bytes = await file.read()
    result = parse_resume(file_bytes, file.filename or "resume.pdf")
    if not result.data:
        raise HTTPException(status_code=400, detail=result.errors or ["Parse failed"])
    return {"success": result.success, "data": result.data.to_dict(), "errors": result.errors}


@app.post("/ats")
def ats_endpoint(body: JobDescriptionRequest) -> dict[str, Any]:
    resume = _resume_from_dict(body.resume)
    result = compute_ats_score(resume, body.job_description)
    return {
        "ats_score": result.ats_score,
        "dimensions": result.dimensions,
        "breakdown": result.breakdown,
    }


@app.post("/skill-gap")
def skill_gap_endpoint(body: JobDescriptionRequest) -> dict[str, Any]:
    resume = _resume_from_dict(body.resume)
    result = analyze_skill_gap(resume, body.job_description)
    return {
        "match_percent": result.match_percent,
        "matched_skills": result.matched_skills,
        "missing_skills": result.missing_skills,
        "recommended_skills": result.recommended_skills,
    }


@app.post("/form-fill")
def form_fill_endpoint(body: FormFillRequest) -> dict[str, str]:
    resume = _resume_from_dict(body.resume)
    questions = [FormQuestion(q) for q in body.questions]
    try:
        return fill_form(resume, questions, job_description=body.job_description)
    except LLMAPIError as exc:
        raise HTTPException(status_code=503, detail=exc.user_message) from exc


@app.post("/summary")
def summary_endpoint(body: SummaryRequest) -> dict[str, str]:
    resume = _resume_from_dict(body.resume)
    try:
        summary = generate_candidate_summary(resume, body.job_description)
        return {"summary": summary}
    except LLMAPIError as exc:
        raise HTTPException(status_code=503, detail=exc.user_message) from exc


@app.post("/improve")
def improve_endpoint(body: RewriteRequest) -> dict[str, Any]:
    resume = _resume_from_dict(body.resume)
    result = improve_resume(resume)
    return {
        "experience": [
            {"original": b.original, "improved": b.improved, "unchanged": b.unchanged}
            for b in result.experience
        ],
        "projects": [
            {"original": b.original, "improved": b.improved, "unchanged": b.unchanged}
            for b in result.projects
        ],
        "errors": result.errors,
    }


@app.post("/analyze-project")
def analyze_project_endpoint(body: ProjectAnalyzeRequest) -> dict[str, Any]:
    result = analyze_github_project(body.github_url)
    return {**result.to_dict(), "errors": result.errors}
