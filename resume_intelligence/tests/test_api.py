"""FastAPI smoke tests."""

from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health_returns_metadata() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["service"] == "resume-intelligence-api"
    assert body["version"] == "1.0.0"
    assert body["llm_provider"] in {"groq", "gemini"}
