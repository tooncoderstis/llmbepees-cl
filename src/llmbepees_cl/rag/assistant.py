"""Asisten BPS: orkestrasi retrieval + tool + jawaban ber-sitasi (Fase 4).

Pada fase ini jawaban konsep masih **ekstraktif** (mengambil kalimat dari dokumen
yang di-retrieve, bukan digenerasi LLM) — supaya pipeline RAG + tool sudah bisa
dijalankan dan diukur. Penggantian ke LLM dilakukan di Fase 5 tanpa mengubah
kontrak `Answer`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from llmbepees_cl.rag.chunking import Passage, chunk_documents
from llmbepees_cl.rag.documents import SAMPLE_DOCS, Document
from llmbepees_cl.rag.embedding import HashingEmbedder, tokenize
from llmbepees_cl.rag.prompt import format_tool_result
from llmbepees_cl.rag.reranker import LexicalReranker
from llmbepees_cl.rag.retrieval import HybridRetriever
from llmbepees_cl.rag.tools import detect_indicator, get_indicator

_YEAR = re.compile(r"\b(19|20)\d{2}\b")
_VALUE_WORDS = (
    "berapa",
    "nilai",
    "jumlah",
    "laju",
    "persentase",
    "persen",
    "besar",
    "angka",
)
ABSTAIN_TEXT = "informasi tidak ditemukan"


@dataclass
class Answer:
    """Jawaban terstruktur dengan jejak audit (sitasi & pemanggilan tool)."""

    text: str
    citations: list[str] = field(default_factory=list)
    tool_calls: list[dict[str, object]] = field(default_factory=list)
    passages: list[Passage] = field(default_factory=list)
    abstained: bool = False


class BPSAssistant:
    """Pipeline RAG + tool untuk pertanyaan konsep dan permintaan angka resmi."""

    def __init__(
        self,
        documents: list[Document] | None = None,
        embedder=None,
        alpha: float = 0.5,
        abstain_threshold: float = 0.25,
        retrieve_k: int = 5,
        rerank_k: int = 3,
    ) -> None:
        self.documents = documents if documents is not None else SAMPLE_DOCS
        self.passages = chunk_documents(self.documents)
        self.retriever = HybridRetriever(
            self.passages, embedder=embedder or HashingEmbedder(), alpha=alpha
        )
        self.reranker = LexicalReranker()
        self.abstain_threshold = abstain_threshold
        self.retrieve_k = retrieve_k
        self.rerank_k = rerank_k

    # -- routing ---------------------------------------------------------------
    def _is_number_question(self, question: str) -> bool:
        if detect_indicator(question) is None:
            return False
        text = question.lower()
        return any(word in text for word in _VALUE_WORDS) or bool(_YEAR.search(question))

    # -- jawaban ---------------------------------------------------------------
    def ask(self, question: str) -> Answer:
        """Jawab satu pertanyaan; abstain bila konteks/tool tidak memadai."""
        if self._is_number_question(question):
            return self._answer_number(question)
        return self._answer_concept(question)

    def _answer_number(self, question: str) -> Answer:
        indicator = detect_indicator(question)
        year = _YEAR.search(question)
        if indicator is None or year is None:
            return Answer(text=ABSTAIN_TEXT, abstained=True)
        record = get_indicator(indicator, region="Indonesia", period=year.group(0))
        if record is None:
            return Answer(text=ABSTAIN_TEXT, abstained=True)
        return Answer(
            text=f"{format_tool_result(record)}.",
            citations=[str(record["source"])],
            tool_calls=[record],
        )

    def _answer_concept(self, question: str) -> Answer:
        candidates = self.retriever.search(question, top_k=self.retrieve_k)
        if not candidates or candidates[0].score < self.abstain_threshold:
            return Answer(text=ABSTAIN_TEXT, abstained=True)
        ranked = self.reranker.rerank(question, candidates, top_k=self.rerank_k)
        best = ranked[0].passage
        # Gerbang abstain: passage terbaik harus cukup menutupi kata penting query
        # (coverage >= 0.5) atau diawali istilah yang ditanyakan (bonus definisi=1.0).
        if ranked[0].match_score < 0.5:
            return Answer(text=ABSTAIN_TEXT, abstained=True)
        sentence = self._best_sentence(best.text, question)
        text = f"{sentence} (Sumber: {best.citation}, bagian {best.section})"
        return Answer(
            text=text,
            citations=[best.citation],
            passages=[c.passage for c in ranked],
        )

    @staticmethod
    def _best_sentence(text: str, question: str) -> str:
        """Pilih kalimat dalam passage yang paling banyak memuat kata pertanyaan."""
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        if len(sentences) <= 1:
            return text.strip()
        q_tokens = set(tokenize(question))
        return max(sentences, key=lambda s: len(q_tokens & set(tokenize(s))))


def build_default_assistant() -> BPSAssistant:
    """Asisten dengan korpus contoh bawaan (`SAMPLE_DOCS`)."""
    return BPSAssistant()
