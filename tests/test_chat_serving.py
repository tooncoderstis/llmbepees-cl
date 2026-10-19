import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from llmbepees_cl.serving.chat import create_chat_app  # noqa: E402


class FakeEngine:
    """Mesin palsu: meniru kontrak `reply(messages) -> str` tanpa memuat model."""

    def __init__(self) -> None:
        self.seen: list[dict[str, str]] = []

    def reply(self, messages: list[dict[str, str]]) -> str:
        self.seen = messages
        return "jawaban uji untuk: " + messages[-1]["content"]


CLIENT = TestClient(create_chat_app(engine=FakeEngine()))


def test_index_returns_html():
    response = CLIENT.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "chat" in response.text.lower()


def test_chat_returns_reply_from_messages():
    response = CLIENT.post(
        "/chat", json={"messages": [{"role": "user", "content": "Apa itu inflasi?"}]}
    )
    assert response.status_code == 200
    assert response.json()["reply"] == "jawaban uji untuk: Apa itu inflasi?"


def test_chat_rejects_empty_messages():
    assert CLIENT.post("/chat", json={"messages": []}).status_code == 422
