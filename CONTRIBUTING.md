# Contributing to Resume Intelligence

Thanks for checking out the project. Contributions are welcome — issues, docs, tests, or small features.

## Getting started

```bash
git clone https://github.com/PranjalManhgaye/Resume_checker.git
cd Resume_checker/resume_intelligence
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
pytest tests/ -v
```

## How to contribute

1. Open an issue or pick an existing one from [Issues](https://github.com/PranjalManhgaye/Resume_checker/issues)
2. Fork the repo and create a branch (`feat/my-thing` or `fix/parser-edge-case`)
3. Keep PRs focused — one concern per PR when possible
4. Run `pytest tests/ -v` before opening a PR
5. Do not commit `.env` or API keys

## Code style

- Type hints on public functions
- Small focused modules under `backend/`, `services/`, `utils/`
- Add tests for new behavior in `tests/`

## Questions?

Open a GitHub issue — happy to help you get unblocked.
