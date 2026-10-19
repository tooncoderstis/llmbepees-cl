"""Chunking: memecah dokumen menjadi bagian (passage) kecil yang bisa dicari.

Chunk yang baik = satu gagasan utuh (mis. satu definisi), bukan sepotong kalimat
acak. Di sini kita memecah per paragraf (dipisah baris kosong) dan, bila terlalu
panjang, per kalimat — sambil menyimpan metadata (judul, tahun, bagian) agar
jawaban bisa disitasi.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from llmbepees_cl.rag.documents import Document

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class Passage:
    """Satu potongan dokumen yang siap diindeks dan dicari."""

    passage_id: str
    doc_id: str
    title: str
    year: int
    section: str
    text: str

    @property
    def citation(self) -> str:
        return f"{self.title} ({self.year})"


def _split_long(text: str, max_chars: int) -> list[str]:
    """Pecah paragraf panjang per kalimat, lalu gabung sampai mendekati `max_chars`."""
    if len(text) <= max_chars:
        return [text]
    parts: list[str] = []
    buffer = ""
    for sentence in _SENTENCE_SPLIT.split(text):
        candidate = f"{buffer} {sentence}".strip()
        if buffer and len(candidate) > max_chars:
            parts.append(buffer)
            buffer = sentence
        else:
            buffer = candidate
    if buffer:
        parts.append(buffer)
    return parts


def chunk_document(document: Document, max_chars: int = 600) -> list[Passage]:
    """Pecah satu dokumen menjadi passage-passage ber-metadata."""
    passages: list[Passage] = []
    for paragraph in document.text.split("\n\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        for piece in _split_long(paragraph, max_chars):
            passages.append(
                Passage(
                    passage_id=f"{document.doc_id}#{len(passages)}",
                    doc_id=document.doc_id,
                    title=document.title,
                    year=document.year,
                    section=document.section,
                    text=piece,
                )
            )
    return passages


def chunk_documents(documents: list[Document], max_chars: int = 600) -> list[Passage]:
    """Pecah daftar dokumen menjadi daftar passage."""
    passages: list[Passage] = []
    for document in documents:
        passages.extend(chunk_document(document, max_chars))
    return passages
