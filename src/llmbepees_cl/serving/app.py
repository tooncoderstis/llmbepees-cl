"""API FastAPI untuk asisten BPS (Fase 5).

Membungkus `BPSAssistant` (Fase 4) menjadi layanan HTTP sederhana. Kontrak jawaban
tetap: `answer`, `citations`, `tool_calls`, `abstained` — sehingga bisa diaudit.
Impor `fastapi` terjadi di modul ini saja (extra `serve`), bukan di paket inti.
"""

from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel, Field

from llmbepees_cl.rag import BPSAssistant, available_indicators, build_default_assistant


class AskRequest(BaseModel):
    question: str = Field(min_length=1, examples=["Apa itu inflasi?"])


class AskResponse(BaseModel):
    answer: str
    citations: list[str]
    tool_calls: list[dict]
    abstained: bool


class HealthResponse(BaseModel):
    status: str
    documents: int
    passages: int


class IndicatorsResponse(BaseModel):
    indicators: list[str]


def create_app(assistant: BPSAssistant | None = None) -> FastAPI:
    """Bangun aplikasi FastAPI; korpus contoh bawaan dipakai bila `assistant` kosong."""
    assistant = assistant or build_default_assistant()
    app = FastAPI(title="llmbepees-cl — asisten BPS", version="0.1.0")

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(
            status="ok", documents=len(assistant.documents), passages=len(assistant.passages)
        )

    @app.post("/ask", response_model=AskResponse)
    def ask(request: AskRequest) -> AskResponse:
        result = assistant.ask(request.question)
        return AskResponse(
            answer=result.text,
            citations=result.citations,
            tool_calls=result.tool_calls,
            abstained=result.abstained,
        )

    @app.get("/indicators", response_model=IndicatorsResponse)
    def indicators() -> IndicatorsResponse:
        return IndicatorsResponse(indicators=available_indicators())

    return app
