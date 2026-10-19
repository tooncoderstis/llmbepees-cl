"""RAG + tool calling (Fase 4): prototipe asisten BPS faktual & ber-sitasi.

Prinsip: konsep/definisi dijawab dari dokumen yang di-retrieve (dengan sitasi),
angka resmi hanya dari tool, dan abstain bila sumber tidak ditemukan. Reusable dan
diuji di `tests/test_rag_*.py`; skrip belajar ada di `04-rag-bps/`.
"""

from llmbepees_cl.rag.assistant import Answer, BPSAssistant, build_default_assistant
from llmbepees_cl.rag.bm25 import BM25Index
from llmbepees_cl.rag.chunking import Passage, chunk_documents
from llmbepees_cl.rag.documents import SAMPLE_DOCS, Document, load_text_files
from llmbepees_cl.rag.embedding import (
    HashingEmbedder,
    SentenceTransformerEmbedder,
    cosine_similarity,
    get_default_embedder,
    tokenize,
)
from llmbepees_cl.rag.prompt import SYSTEM_PROMPT, build_context, build_prompt
from llmbepees_cl.rag.reranker import LexicalReranker
from llmbepees_cl.rag.retrieval import HybridRetriever, ScoredPassage
from llmbepees_cl.rag.tools import all_records, available_indicators, get_indicator

__all__ = [
    "SAMPLE_DOCS",
    "SYSTEM_PROMPT",
    "Answer",
    "BM25Index",
    "BPSAssistant",
    "Document",
    "HashingEmbedder",
    "HybridRetriever",
    "LexicalReranker",
    "Passage",
    "ScoredPassage",
    "SentenceTransformerEmbedder",
    "all_records",
    "available_indicators",
    "build_context",
    "build_default_assistant",
    "build_prompt",
    "chunk_documents",
    "cosine_similarity",
    "get_default_embedder",
    "get_indicator",
    "load_text_files",
    "tokenize",
]
