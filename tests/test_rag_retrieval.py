import numpy as np

from llmbepees_cl.rag import (
    BM25Index,
    HashingEmbedder,
    HybridRetriever,
    LexicalReranker,
    chunk_documents,
    cosine_similarity,
    tokenize,
)
from llmbepees_cl.rag.documents import SAMPLE_DOCS

PASSAGES = chunk_documents(SAMPLE_DOCS)


def test_tokenize_lowercases_and_splits():
    assert tokenize("Indeks Harga, Konsumen!") == ["indeks", "harga", "konsumen"]


def test_hashing_embedder_is_deterministic_and_normalized():
    embedder = HashingEmbedder(dim=64)
    a = embedder.encode(["inflasi harga barang"])[0]
    b = HashingEmbedder(dim=64).encode(["inflasi harga barang"])[0]
    assert np.allclose(a, b)
    assert abs(np.linalg.norm(a) - 1.0) < 1e-9


def test_cosine_similarity_matches_identical():
    embedder = HashingEmbedder(dim=64)
    matrix = embedder.encode(["inflasi harga", "sensus penduduk"])
    query = embedder.encode(["inflasi harga"])[0]
    sims = cosine_similarity(matrix, query)
    assert sims[0] > sims[1]


def test_bm25_ranks_relevant_passage_first():
    index = BM25Index([p.text for p in PASSAGES])
    scores = index.scores("inflasi")
    best = PASSAGES[int(np.argmax(scores))]
    assert "Inflasi" in best.text


def test_hybrid_retriever_returns_relevant():
    retriever = HybridRetriever(PASSAGES)
    results = retriever.search("Apa itu inflasi?", top_k=3)
    assert results
    assert "Inflasi" in results[0].passage.text
    assert results[0].score >= results[-1].score


def test_reranker_prefers_covered_passage():
    retriever = HybridRetriever(PASSAGES)
    reranker = LexicalReranker()
    ranked = reranker.rerank(
        "Jelaskan konsep sampel.", retriever.search("sampel", top_k=5), top_k=3
    )
    assert "Sampel" in ranked[0].passage.text
