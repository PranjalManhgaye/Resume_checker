# Data & Evaluation Notes

Honest reference for project submissions, demos, and reviewers.

## Datasets

| Asset | Location | Notes |
|-------|----------|-------|
| Sample resumes | `resume_intelligence/data/samples/` | 2 synthetic fixtures (PDF + DOCX), ~77 KB total |
| Training corpus | — | None — no model training in this repo |
| Job descriptions | User-provided at runtime | Pasted in UI or sent via API |

Samples are for parser/ATS smoke tests only, not a benchmark suite.

## What we actually measure

| Check | How | Current result |
|-------|-----|----------------|
| Unit / integration tests | `pytest tests/ -v` | 26 tests, all passing |
| Parser smoke (samples) | Manual + tests on 2 fixtures | Email/contact extracted on both |
| ATS dimensions | Heuristics + scikit-learn TF-IDF | Deterministic per resume + JD |
| LLM features | Groq or Gemini API | Depends on provider quota/latency |

We do **not** claim a published accuracy % on real hiring data — that would need a labeled corpus we don't have.

## ML / NLP stack

- **Primary scoring:** scikit-learn `TfidfVectorizer`, regex/heuristic rules
- **Embeddings:** Sentence Transformers loaded for warmup; ATS scoring uses TF-IDF, not cosine on embeddings
- **LLM:** Groq (default) or Google Gemini via `services/llm_client.py`

## Reproducing results

```bash
cd resume_intelligence
source .venv/bin/activate
pytest tests/ -v
streamlit run frontend/app.py
```

Set `LLM_PROVIDER` and API keys in `.env` for AI features.

## Suggested future eval (not implemented)

- Larger resume fixture set with expected parse fields
- Golden-file ATS breakdown snapshots
- Latency benchmarks for `/parse` and `/ats`

See GitHub issues for tracking.
