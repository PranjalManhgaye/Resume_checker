# Resume Intelligence Platform

AI-powered resume analysis: parsing, ATS scoring, skill gap analysis, form autofill, and more.

## Setup

```bash
cd resume_intelligence
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Recommended on slow networks — CPU-only PyTorch before sentence-transformers:
# pip install torch --index-url https://download.pytorch.org/whl/cpu
# pip install sentence-transformers

cp .env.example .env        # then add your GROQ_API_KEY or GEMINI_API_KEY
```

## Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `LLM_PROVIDER` | No | `groq` (default, fast) or `gemini` |
| `GROQ_API_KEY` | If using Groq | API key from [Groq Console](https://console.groq.com) |
| `GROQ_MODEL` | No | Default: `llama-3.1-8b-instant` |
| `GEMINI_API_KEY` | If using Gemini | Google Gemini API key from [AI Studio](https://aistudio.google.com/apikey) |
| `GEMINI_MODEL` | No | Default: `gemini-flash-latest` |
| `GITHUB_TOKEN` | No | Higher GitHub API rate limits for project analyzer |

## Run Streamlit UI

```bash
source .venv/bin/activate
streamlit run frontend/app.py
```

## Run FastAPI

```bash
source .venv/bin/activate
uvicorn api.main:app --reload --port 8000
```

API docs: http://localhost:8000/docs

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/parse` | POST | Upload resume file |
| `/ats` | POST | ATS score |
| `/skill-gap` | POST | Skill gap analysis |
| `/form-fill` | POST | Form autofill |
| `/summary` | POST | Candidate summary |
| `/improve` | POST | Resume bullet improvements |
| `/analyze-project` | POST | GitHub project analysis |

## Test LLM API

```bash
cd resume_intelligence
source .venv/bin/activate
python -c "from services.llm_client import get_llm_client; print(get_llm_client().generate_text('Say hello in 3 words'))"
```

## Tests

```bash
pytest tests/ -v
```

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `python` not found | Use `python3` or activate `.venv` first |
| `403 PERMISSION_DENIED` | Invalid or blocked API key — create new key in AI Studio |
| `503 UNAVAILABLE` | Model overloaded — retries happen automatically; try `GEMINI_MODEL_FALLBACK` |
| Slow first ATS run | Embedding model warms up on app start; first run may still take ~30s |
| Resume Improvement unchanged | API failed (check terminal) or bullets lack detail — add description bullets |

## Project structure

```
resume_intelligence/
├── frontend/app.py          # Streamlit UI
├── api/main.py              # FastAPI REST API
├── backend/                 # Feature modules
├── models/embeddings.py     # Sentence Transformers wrapper
├── services/gemini_client.py
├── utils/
├── data/samples/
└── tests/
```
