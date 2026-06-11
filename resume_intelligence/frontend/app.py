"""Resume Intelligence Platform — Streamlit frontend."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

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
from models.embeddings import warmup_embedding_model
from models.resume import ResumeData
from services.llm_client import LLMAPIError
from utils.cache_keys import make_cache_key
from utils.file_utils import read_binary_upload, validate_resume_file

st.set_page_config(
    page_title="Resume Intelligence",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

PAGES = [
    "Resume Upload",
    "ATS Analysis",
    "Skill Gap Analysis",
    "Form Autofill",
    "Resume Improvement",
    "Candidate Summary",
    "Project Analyzer",
]


@st.cache_resource
def _load_embedding_model() -> bool:
    """Warm up SentenceTransformer on first app load."""
    warmup_embedding_model()
    return True


def init_session_state() -> None:
    defaults = {
        "resume": None,
        "resume_dict": None,
        "job_description": "",
        "ats_result": None,
        "skill_gap": None,
        "form_answers": None,
        "rewrite_result": None,
        "candidate_summary": "",
        "project_analysis": None,
        "ats_cache_key": "",
        "skill_gap_cache_key": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def get_resume() -> ResumeData | None:
    data = st.session_state.get("resume")
    if isinstance(data, ResumeData):
        return data
    return None


def require_resume() -> ResumeData | None:
    resume = get_resume()
    if resume is None:
        st.warning("Upload and parse a resume on the **Resume Upload** page first.")
    return resume


def show_ai_error(exc: Exception) -> None:
    """Display user-friendly AI error messages."""
    if isinstance(exc, LLMAPIError):
        st.error(exc.user_message)
    else:
        st.error(f"Request failed: {exc}")


def render_sidebar() -> str:
    st.sidebar.title("Resume Intelligence")
    st.sidebar.caption("AI-powered resume analysis platform")
    page = st.sidebar.radio("Navigate", PAGES, label_visibility="collapsed")

    if st.session_state.get("resume_dict"):
        st.sidebar.divider()
        st.sidebar.success("Resume loaded")
        name = st.session_state["resume_dict"].get("name", "Unknown")
        st.sidebar.write(f"**{name}**")

    return page


def page_upload() -> None:
    st.header("Resume Upload")
    st.write("Upload a PDF or DOCX resume to parse and structure its content.")

    uploaded = st.file_uploader("Choose a resume file", type=["pdf", "docx"])

    if uploaded is not None:
        file_bytes = read_binary_upload(uploaded)
        validation = validate_resume_file(uploaded.name, file_bytes)

        if not validation.valid:
            st.error(validation.error)
            return

        if st.button("Parse Resume", type="primary"):
            with st.status("Parsing resume...", expanded=True) as status:
                st.write("Extracting text and detecting sections...")
                result = parse_resume(file_bytes, uploaded.name)

                if result.errors:
                    for err in result.errors:
                        st.warning(err)

                if result.data:
                    st.session_state["resume"] = result.data
                    st.session_state["resume_dict"] = result.data.to_dict()
                    st.session_state["ats_result"] = None
                    st.session_state["skill_gap"] = None
                    status.update(label="Parse complete", state="complete")

                    if result.data.parse_warnings:
                        for w in result.data.parse_warnings:
                            st.info(w)
                else:
                    status.update(label="Parse failed", state="error")
                    st.error("Could not parse resume.")

    if st.session_state.get("resume_dict"):
        st.subheader("Parsed Resume")
        st.json(st.session_state["resume_dict"])

        st.download_button(
            label="Download JSON",
            data=json.dumps(st.session_state["resume_dict"], indent=2),
            file_name="resume_parsed.json",
            mime="application/json",
        )


def page_ats() -> None:
    st.header("ATS Analysis")
    resume = require_resume()
    if not resume:
        return

    job_description = st.text_area(
        "Job Description",
        value=st.session_state.get("job_description", ""),
        height=200,
        placeholder="Paste the job description here...",
    )

    if st.button("Run ATS Analysis", type="primary"):
        st.session_state["job_description"] = job_description
        cache_key = make_cache_key("ats", resume.email, job_description)

        if cache_key == st.session_state.get("ats_cache_key") and st.session_state.get("ats_result"):
            st.info("Showing cached results. Change the job description to recompute.")
        else:
            with st.spinner("Computing ATS score..."):
                result = compute_ats_score(resume, job_description)
                st.session_state["ats_result"] = result
                st.session_state["ats_cache_key"] = cache_key

    result = st.session_state.get("ats_result")
    if result:
        st.metric("Overall ATS Score", f"{result.ats_score}%")

        st.subheader("8-Dimension Scores")
        labels = result.breakdown.get("dimension_labels", {})
        dims = result.dimensions

        row1 = st.columns(4)
        row2 = st.columns(4)
        dim_items = list(dims.items())
        for i, (key, score) in enumerate(dim_items):
            col = row1[i] if i < 4 else row2[i - 4]
            col.metric(labels.get(key, key.replace("_", " ").title()), f"{score}%")

        if result.breakdown.get("tfidf_keywords"):
            with st.expander("TF-IDF Keyword Alignment"):
                kw = result.breakdown["tfidf_keywords"]
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Matched keywords**")
                    for term in kw.get("matched_keywords", []):
                        st.success(term)
                with c2:
                    st.markdown("**Missing keywords**")
                    for term in kw.get("missing_keywords", []):
                        st.error(term)

        st.subheader("Full Breakdown")
        st.json(
            {
                "ats_score": result.ats_score,
                "dimensions": result.dimensions,
                "breakdown": result.breakdown,
            }
        )

        st.download_button(
            "Download ATS Report",
            data=json.dumps(
                {
                    "ats_score": result.ats_score,
                    "dimensions": result.dimensions,
                    "breakdown": result.breakdown,
                },
                indent=2,
            ),
            file_name="ats_report.json",
            mime="application/json",
        )


def page_skill_gap() -> None:
    st.header("Skill Gap Analysis")
    resume = require_resume()
    if not resume:
        return

    job_description = st.text_area(
        "Job Description",
        value=st.session_state.get("job_description", ""),
        height=200,
    )

    if st.button("Analyze Skill Gap", type="primary"):
        st.session_state["job_description"] = job_description
        cache_key = make_cache_key("skills", resume.email, job_description)

        if cache_key == st.session_state.get("skill_gap_cache_key") and st.session_state.get("skill_gap"):
            st.info("Showing cached results. Change the job description to recompute.")
        else:
            with st.spinner("Analyzing skills..."):
                st.session_state["skill_gap"] = analyze_skill_gap(resume, job_description)
                st.session_state["skill_gap_cache_key"] = cache_key

    gap = st.session_state.get("skill_gap")
    if gap:
        st.metric("Match Percentage", f"{gap.match_percent}%")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("Matched Skills")
            if gap.matched_skills:
                for s in gap.matched_skills:
                    st.success(s)
            else:
                st.write("None")
        with col2:
            st.subheader("Missing Skills")
            if gap.missing_skills:
                for s in gap.missing_skills:
                    st.error(s)
            else:
                st.write("None")
        with col3:
            st.subheader("Recommended Skills")
            if gap.recommended_skills:
                for s in gap.recommended_skills:
                    st.info(s)
            else:
                st.write("None")

        st.download_button(
            "Download Skill Gap Report",
            data=json.dumps(
                {
                    "match_percent": gap.match_percent,
                    "matched_skills": gap.matched_skills,
                    "missing_skills": gap.missing_skills,
                    "recommended_skills": gap.recommended_skills,
                },
                indent=2,
            ),
            file_name="skill_gap.json",
            mime="application/json",
        )


def page_form_autofill() -> None:
    st.header("Form Autofill")
    resume = require_resume()
    if not resume:
        return

    st.write("Add application form questions below.")

    if "form_questions" not in st.session_state:
        st.session_state["form_questions"] = [
            "What is your full name?",
            "What is your email address?",
            "What is your CGPA?",
            "Describe your work experience.",
        ]

    questions_text = st.text_area(
        "Questions (one per line)",
        value="\n".join(st.session_state["form_questions"]),
        height=150,
    )

    if st.button("Generate Answers", type="primary"):
        lines = [q.strip() for q in questions_text.splitlines() if q.strip()]
        st.session_state["form_questions"] = lines
        questions = [FormQuestion(q) for q in lines]

        with st.spinner("Generating answers..."):
            try:
                answers = fill_form(resume, questions)
                st.session_state["form_answers"] = answers
            except Exception as exc:
                show_ai_error(exc)

    answers = st.session_state.get("form_answers")
    if answers:
        st.subheader("Generated Answers")
        for question, answer in answers.items():
            st.markdown(f"**{question}**")
            st.write(answer)
            st.divider()

        st.download_button(
            "Download Answers",
            data=json.dumps(answers, indent=2),
            file_name="form_answers.json",
            mime="application/json",
        )


def page_resume_improvement() -> None:
    st.header("Resume Improvement")
    resume = require_resume()
    if not resume:
        return

    st.write("Generate improved bullet points while preserving factual accuracy.")

    if st.button("Improve Resume Bullets", type="primary"):
        with st.spinner("Improving resume content..."):
            try:
                result = improve_resume(resume)
                st.session_state["rewrite_result"] = result
            except Exception as exc:
                show_ai_error(exc)

    result = st.session_state.get("rewrite_result")
    if result:
        if result.errors:
            for err in result.errors:
                st.warning(err)

        if result.experience:
            st.subheader("Experience")
            for bullet in result.experience:
                st.markdown("**Original:**")
                st.write(bullet.original)
                st.markdown("**Improved:**")
                if bullet.unchanged:
                    st.warning(bullet.improved)
                    st.caption("Unchanged — add more detail to this bullet for better improvements.")
                else:
                    st.success(bullet.improved)
                st.divider()

        if result.projects:
            st.subheader("Projects")
            for bullet in result.projects:
                st.markdown("**Original:**")
                st.write(bullet.original)
                st.markdown("**Improved:**")
                if bullet.unchanged:
                    st.warning(bullet.improved)
                    st.caption("Unchanged — add more detail to this bullet for better improvements.")
                else:
                    st.success(bullet.improved)
                st.divider()


def page_candidate_summary() -> None:
    st.header("Candidate Summary")
    resume = require_resume()
    if not resume:
        return

    job_description = st.text_area(
        "Job Description (optional)",
        value=st.session_state.get("job_description", ""),
        height=120,
    )

    if st.button("Generate Summary", type="primary"):
        with st.spinner("Generating recruiter-style summary..."):
            try:
                summary = generate_candidate_summary(resume, job_description)
                st.session_state["candidate_summary"] = summary
            except Exception as exc:
                show_ai_error(exc)

    summary = st.session_state.get("candidate_summary", "")
    if summary:
        st.subheader("Recruiter Summary")
        st.info(summary)

        st.download_button(
            "Download Summary",
            data=summary,
            file_name="candidate_summary.txt",
            mime="text/plain",
        )


def page_project_analyzer() -> None:
    st.header("Project Analyzer")
    st.write("Analyze a GitHub repository README for technologies, domain, and metrics.")

    github_url = st.text_input(
        "GitHub Repository URL",
        placeholder="https://github.com/owner/repo",
    )

    if st.button("Analyze Project", type="primary") and github_url:
        with st.spinner("Fetching and analyzing repository..."):
            try:
                analysis = analyze_github_project(github_url)
                st.session_state["project_analysis"] = analysis
            except Exception as exc:
                show_ai_error(exc)

    analysis = st.session_state.get("project_analysis")
    if analysis:
        if analysis.errors:
            for err in analysis.errors:
                st.warning(err)

        st.json(analysis.to_dict())

        st.download_button(
            "Download Analysis",
            data=json.dumps(analysis.to_dict(), indent=2),
            file_name="project_analysis.json",
            mime="application/json",
        )


PAGE_RENDERERS = {
    "Resume Upload": page_upload,
    "ATS Analysis": page_ats,
    "Skill Gap Analysis": page_skill_gap,
    "Form Autofill": page_form_autofill,
    "Resume Improvement": page_resume_improvement,
    "Candidate Summary": page_candidate_summary,
    "Project Analyzer": page_project_analyzer,
}


def main() -> None:
    init_session_state()
    _load_embedding_model()
    page = render_sidebar()
    PAGE_RENDERERS[page]()


if __name__ == "__main__":
    main()
