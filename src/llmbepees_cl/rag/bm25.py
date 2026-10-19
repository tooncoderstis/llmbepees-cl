"""BM25 dari nol — pencarian berbasis kata kunci (leksikal).

BM25 menilai seberapa cocok sebuah dokumen dengan query dengan mempertimbangkan
frekuensi kata di dokumen (tf) dan kelangkaan kata di seluruh korpus (idf), plus
normalisasi panjang dokumen. Ini pelengkap pencarian vektor: kuat untuk istilah
persis, lemah untuk sinonim.
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np

from llmbepees_cl.rag.embedding import tokenize


class BM25Index:
    """Indeks BM25 atas daftar teks (dokumen/passage)."""

    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.doc_tokens: list[list[str]] = [tokenize(t) for t in texts]
        self.doc_len = np.array([len(t) for t in self.doc_tokens], dtype=np.float64)
        self.avgdl = float(self.doc_len.mean()) if len(texts) else 0.0
        self.n_docs = len(texts)

        df: Counter[str] = Counter()
        for tokens in self.doc_tokens:
            df.update(set(tokens))
        self.idf: dict[str, float] = {
            term: math.log((self.n_docs - freq + 0.5) / (freq + 0.5) + 1.0)
            for term, freq in df.items()
        }

    def scores(self, query: str) -> np.ndarray:
        """Skor BM25 tiap dokumen terhadap `query`."""
        scores = np.zeros(self.n_docs, dtype=np.float64)
        if self.n_docs == 0:
            return scores
        for term in tokenize(query):
            if term not in self.idf:
                continue
            idf = self.idf[term]
            for i, tokens in enumerate(self.doc_tokens):
                tf = tokens.count(term)
                if tf == 0:
                    continue
                denom = tf + self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avgdl)
                scores[i] += idf * tf * (self.k1 + 1) / denom
        return scores
