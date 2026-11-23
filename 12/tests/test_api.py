"""Автоматичні тести API (pytest + TestClient). LLM підміняється заглушкою — тести швидкі й детерміновані."""
import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import main                                  # noqa: E402
from app.core import LLMError, GraphUnavailable        # noqa: E402


def fake_llm(messages, **kw):
    return {"text": "Допустимий діапазон +12 В — 11,40–12,60 В [KB-03:s2].", "tokens_in": 900, "tokens_out": 30,
            "latency_s": 0.01, "cost_usd": 0.0002}


@pytest.fixture(scope="module")
def client():
    main.system().llm = fake_llm
    return TestClient(main.app)


def test_success(client):
    r = client.post("/ask", json={"question": "Яка допустима напруга на лінії +12 В?"})
    assert r.status_code == 200
    d = r.json()
    assert d["citations"][0]["id"] == "KB-03:s2" and "X-Request-ID" in r.headers


def test_bad_payload(client):
    assert client.post("/ask", json={"q": "нема поля question"}).status_code == 422
    assert client.post("/ask", json={"question": "   "}).status_code == 422
    assert client.post("/ask", json={"question": "ok?", "k": 50}).status_code == 422


def test_too_long(client):
    r = client.post("/ask", json={"question": "а" * 501})
    assert r.status_code == 422 and "500" in r.text


def test_llm_timeout(client, monkeypatch):
    def boom(*a, **k): raise LLMError("ReadTimeout")
    monkeypatch.setattr(main.system(), "llm", boom)
    r = client.post("/ask", json={"question": "Що означає код 0x0000007B?"})
    assert r.status_code == 504 and r.json()["error"]["code"] == "LLM_TIMEOUT"


def test_graph_down(client, monkeypatch):
    def down(q): raise GraphUnavailable("ServiceUnavailable")
    monkeypatch.setattr(main.system(), "graph_context", down)
    r = client.post("/ask", json={"question": "ПК пищить при старті"})
    assert r.status_code == 503 and r.json()["error"]["code"] == "GRAPH_UNAVAILABLE"


def test_response_format(client):
    d = client.post("/ask", json={"question": "Комп'ютер пищить і синій екран", "k": 3}).json()
    for key in ("request_id", "answer", "refused", "citations", "graph_edges", "concepts", "tokens_in", "latency_ms",
                "model", "prompt_version"):
        assert key in d
    assert isinstance(d["graph_edges"], list) and len(d["chunks"]) <= 3


def test_sources_and_health(client):
    assert client.get("/sources/KB-05:s1").json()["doc"] == "KB-05"
    assert client.get("/sources/XX-99:s0").status_code == 404
    assert client.get("/health").json()["components"]["index"]["ok"] is True
