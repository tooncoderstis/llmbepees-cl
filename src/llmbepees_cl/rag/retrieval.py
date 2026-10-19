"""Pencarian hybrid: gabungkan BM25 (leksikal) + embedding (semantik).

Kenapa hybrid? BM25 unggul menangkap istilah persis (nama konsep, singkatan),
sementara vektor unggul pada parafrase/sinonim. Menggabungkan skor keduanya
(weighted fusion) biasanya lebih tahan daripada salah satu saja.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from llmbepees_cl.rag.bm25 import BM25Index
from llmbepees_cl.rag.chunking import Passage
from llmbepees_cl.rag.embedding import HashingEmbedder, cosine_similarity


@dataclass(frozen=True)
class ScoredPassage:
    """Passage dengan skor gabungan dan komponennya (untuk transparansi).

    `match_score` diisi oleh reranker (coverage kata query + bonus definisi);
    default 0 sebelum reranking.
    """

    passage: Passage
    score: float
    bm25: float
    vector: float
    match_score: float = 0.0


class HybridRetriever:
    """Indeks hybrid atas daftar passage; panggil `search(query)`."""

    def __init__(self, passages: list[Passage], embedder=None, alpha: float = 0.5) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha harus di antara 0 dan 1")
        self.passages = list(passages)
        self.alpha = alpha
        self.embedder = embedder or HashingEmbedder()
        self.bm25 = BM25Index([p.text for p in self.passages])
        self.matrix = self.embedder.encode([p.text for p in self.passages])

    def search(self, query: str, top_k: int = 5) -> list[ScoredPassage]:
        """Kembalikan `top_k` passage paling relevan (skor menurun)."""
        if not self.passages:
            return []
        bm25_scores = self.bm25.scores(query)
        bm25_max = bm25_scores.max()
        bm25_norm = bm25_scores / bm25_max if bm25_max > 0 else bm25_scores

        vector_scores = cosine_similarity(self.matrix, self.embedder.encode([query])[0])
        combined = self.alpha * vector_scores + (1 - self.alpha) * bm25_norm

        order = np.argsort(combined)[::-1][:top_k]
        return [
            ScoredPassage(
                passage=self.passages[i],
                score=float(combined[i]),
                bm25=float(bm25_norm[i]),
                vector=float(vector_scores[i]),
            )
            for i in order
        ]
