"""Reranker ringan: menyusun ulang kandidat agar yang paling relevan di atas.

Versi awal ini **leksikal** dan sadar-definisi:
- **coverage**: berapa banyak kata penting query muncul di passage;
- **bonus definisi**: passage yang diawali istilah yang dicari (pola "X adalah ...")
  dinaikkan, sehingga "Apa itu populasi?" memilih passage "Populasi adalah ..."
  alih-alih passage lain yang hanya menyebut kata "populasi".

Reranker cross-encoder (lebih akurat, lebih mahal) bisa ditambahkan kemudian.
"""

from __future__ import annotations

from dataclasses import replace

from llmbepees_cl.rag.embedding import tokenize
from llmbepees_cl.rag.retrieval import ScoredPassage

_STOPWORDS = {
    "apa",
    "apakah",
    "itu",
    "yang",
    "adalah",
    "dan",
    "atau",
    "di",
    "ke",
    "dari",
    "untuk",
    "pada",
    "dalam",
    "dengan",
    "serta",
    "ini",
    "the",
    "a",
    "an",
    "of",
    "is",
    "berapa",
    "jelaskan",
    "sebutkan",
    "sebut",
    "siapa",
    "kapan",
    "dimana",
    "bagaimana",
    "mengapa",
    "kenapa",
    "tentang",
    "arti",
    "pengertian",
    "definisi",
    "konsep",
    "maksud",
    "dimaksud",
}


def content_tokens(text: str) -> list[str]:
    """Token bermakna (tanpa kata tanya/umum) untuk menilai coverage."""
    return [t for t in tokenize(text) if t not in _STOPWORDS]


def _definition_head(text: str, max_tokens: int = 4) -> set[str]:
    """Istilah di awal passage (sebelum kata 'adalah') sebagai penanda subjek."""
    tokens = tokenize(text)
    head = tokens[: tokens.index("adalah")] if "adalah" in tokens else tokens[:max_tokens]
    return {t for t in head if t not in _STOPWORDS}


class LexicalReranker:
    """Urutkan ulang kandidat berdasarkan coverage + bonus definisi."""

    def rerank(
        self, query: str, candidates: list[ScoredPassage], top_k: int = 3
    ) -> list[ScoredPassage]:
        q_tokens = set(content_tokens(query))
        if not q_tokens:
            return candidates[:top_k]

        scored: list[ScoredPassage] = []
        for candidate in candidates:
            doc_tokens = set(content_tokens(candidate.passage.text))
            coverage = len(q_tokens & doc_tokens) / len(q_tokens)
            head = _definition_head(candidate.passage.text)
            bonus = 1.0 if head and head <= q_tokens else 0.0
            scored.append(replace(candidate, match_score=coverage + bonus))

        ranked = sorted(scored, key=lambda c: (c.match_score, c.score), reverse=True)
        return ranked[:top_k]
