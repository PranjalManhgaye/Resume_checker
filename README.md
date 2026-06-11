# Resume Intelligence Platform

![Python](https://img.shields.io/badge/python-3.12+-blue.svg)
![Tests](https://img.shields.io/badge/tests-26%20passing-brightgreen.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-REST-009688.svg)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)

An AI-powered resume analysis platform that goes beyond simple ATS checking. Upload a resume, match it against job descriptions, identify skill gaps, auto-fill application forms, and generate recruiter-ready summaries.

Built with Python, Streamlit, Sentence Transformers, and pluggable LLM backends (Groq / Gemini).

---

## Features

| Feature | Description |
|---------|-------------|
| **Resume Parser** | Extract structured data from PDF and DOCX (PyMuPDF + pdfplumber fallback) |
| **ATS Scoring** | 8-dimension score: impact, quantification, TF-IDF keywords, sections, verbs, skills, formatting, role-fit |
| **Skill Gap Analysis** | Matched, missing, and recommended skills from job descriptions |
| **Form Autofill** | Deterministic + AI answers for application questions |
| **Resume Improvement** | Rephrase bullets without inventing facts |
| **Candidate Summary** | Recruiter-style 2–3 sentence summaries |
| **Project Analyzer** | GitHub README → structured project metadata |
| **REST API** | FastAPI endpoints for all core features |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit UI  │  FastAPI REST           │
├─────────────────────────────────────────────────────────────┤
│  parser │ ats │ matcher │ form_filler │ rewriter │ ...    │
├─────────────────────────────────────────────────────────────┤
│  Sentence Transformers (embeddings)  │  Groq / Gemini LLM  │
└─────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

- **Frontend:** Streamlit
- **API:** FastAPI + Uvicorn
- **NLP:** Sentence Transformers (`all-MiniLM-L6-v2`), cosine similarity
- **Parsing:** PyMuPDF, pdfplumber, python-docx
- **LLM:** Groq (default, fast) or Google Gemini
- **Testing:** pytest

---

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/PranjalManhgaye/Resume_checker.git
cd Resume_checker/resume_intelligence

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Optional — faster CPU-only PyTorch on slow networks:
# pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### 2. Configure environment

```bash
cp .env.example .env
```

Edit `.env` with your API keys:

```env
LLM_PROVIDER=groq
GROQ_API_KEY=your_groq_key_here
```

Get a free Groq key at [console.groq.com](https://console.groq.com).

### 3. Run the app

```bash
streamlit run frontend/app.py
```

Open http://localhost:8501

### 4. Run tests

```bash
pytest tests/ -v
```

---

## API Server

```bash
uvicorn api.main:app --reload --port 8000
```

Interactive docs: http://localhost:8000/docs

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/parse` | POST | Upload resume (PDF/DOCX) |
| `/ats` | POST | ATS score + breakdown |
| `/skill-gap` | POST | Skill gap analysis |
| `/form-fill` | POST | Application form autofill |
| `/summary` | POST | Candidate summary |
| `/improve` | POST | Resume bullet improvements |
| `/analyze-project` | POST | GitHub project analysis |

---

## Project Structure

```
Resume_checker/
├── README.md
├── LICENSE
└── resume_intelligence/
    ├── frontend/app.py       # Streamlit UI
    ├── api/main.py           # FastAPI REST API
    ├── backend/              # Core feature modules
    ├── models/               # Data models + embeddings
    ├── services/             # LLM clients (Groq, Gemini)
    ├── utils/
    ├── data/samples/         # Test resume fixtures
    └── tests/                # pytest suite
```

---

## ATS Scoring Algorithm

Scores resumes across **8 dimensions** (12.5% weight each):

| Dimension | Method |
|-----------|--------|
| Impact Language | Impact-word presence in bullets |
| Quantification | Numbers, metrics, percentages detected |
| ATS Keyword Alignment | TF-IDF keyword extraction + resume coverage |
| Section Completeness | Education, skills, experience, projects, contact |
| Action Verb Density | Strong action verb frequency |
| Skill Relevance | Required skill overlap with job description |
| Formatting | Contact info, length, structure heuristics |
| Role-Fit | TF-IDF cosine similarity (resume vs. job description) |

Keyword alignment and role-fit use [scikit-learn TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html) to map job-description terms against resume content.

For formulas, weights, TF-IDF params, and limitations see **[ALGORITHM.md](ALGORITHM.md)**.

---

## Design Principles

- **Truthful AI** — never invent metrics, CGPA, or achievements
- **Graceful degradation** — partial parse results with warnings on failure
- **Modular backend** — each feature is an independent, testable module
- **Provider-agnostic LLM** — switch between Groq and Gemini via `.env`

---

## Author

**Pranjal Manhgaye**

- GitHub: [@PranjalManhgaye](https://github.com/PranjalManhgaye)

---

## License

MIT — see [LICENSE](LICENSE).
