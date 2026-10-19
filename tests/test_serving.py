import pytest

from llmbepees_cl.rag import build_default_assistant
from llmbepees_cl.serving import benchmark_ask

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from llmbepees_cl.serving.app import create_app  # noqa: E402

CLIENT = TestClient(create_app())


def test_health():
    response = CLIENT.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["passages"] > 0


def test_ask_concept_cites_source():
    response = CLIENT.post("/ask", json={"question": "Apa itu inflasi?"})
    assert response.status_code == 200
    body = response.json()
    assert body["abstained"] is False
    assert body["citations"]
    assert body["tool_calls"] == []


def test_ask_number_uses_tool():
    response = CLIENT.post("/ask", json={"question": "Berapa inflasi Indonesia 2023?"})
    body = response.json()
    assert len(body["tool_calls"]) == 1
    assert body["tool_calls"][0]["period"] == "2023"


def test_ask_abstains_out_of_corpus():
    response = CLIENT.post("/ask", json={"question": "Apa warna bendera BPS?"})
    assert response.json()["abstained"] is True


def test_ask_rejects_empty_question():
    assert CLIENT.post("/ask", json={"question": ""}).status_code == 422


def test_indicators_endpoint():
    response = CLIENT.get("/indicators")
    assert "inflasi" in response.json()["indicators"]


def test_benchmark_returns_all_samples():
    result = benchmark_ask(build_default_assistant(), ["Apa itu inflasi?"], repeats=3, warmup=1)
    assert result.n == 3
    assert result.p50_ms > 0
    assert result.p95_ms >= result.p50_ms
