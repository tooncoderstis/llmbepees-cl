"""Embedding teks: mengubah teks menjadi vektor agar bisa dicari secara semantik.

Dua pilihan:
- `HashingEmbedder` (bawaan): *feature hashing* deterministik, tanpa unduhan model.
  Cepat dan ramah RAM 8 GB, dipakai untuk tes/CI dan sebagai fallback.
- `SentenceTransformerEmbedder` (opsional): embedding **multibahasa** sungguhan
  via `sentence-transformers` (unduh model ~0,5 GB; butuh izin pengguna).

`get_default_embedder` memakai yang multibahasa bila terpasang, kalau tidak
kembali ke hashing.
"""

from __future__ import annotations

import hashlib
import re

import numpy as np

_WORD = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Pecah teks menjadi token huruf kecil (alfanumerik, sadar-Unicode)."""
    return _WORD.findall(text.lower())


class HashingEmbedder:
    """Embedding bag-of-words via hashing fitur (tanpa model, deterministik)."""

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    @staticmethod
    def _bucket(token: str, dim: int) -> int:
        digest = hashlib.md5(token.encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "little") % dim

    def encode_one(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float64)
        for token in tokenize(text):
            vec[self._bucket(token, self.dim)] += 1.0
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float64)
        return np.vstack([self.encode_one(t) for t in texts])


class SentenceTransformerEmbedder:
    """Embedding multibahasa via `sentence-transformers` (import ditunda)."""

    def __init__(
        self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    ):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1), dtype=np.float64)
        return np.asarray(self.model.encode(texts, normalize_embeddings=True), dtype=np.float64)


def get_default_embedder(prefer_sentence_transformer: bool = True):
    """Pilih embedder: multibahasa bila tersedia, jika tidak hashing."""
    if prefer_sentence_transformer:
        try:
            return SentenceTransformerEmbedder()
        except ImportError:
            pass
    return HashingEmbedder()


def cosine_similarity(matrix: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Kemiripan kosinus antara tiap baris `matrix` dan satu vektor `query`."""
    if matrix.size == 0:
        return np.zeros(0, dtype=np.float64)
    denom = np.linalg.norm(matrix, axis=1) * (np.linalg.norm(query) or 1.0)
    denom[denom == 0] = 1.0
    return (matrix @ query) / denom
