"""Skrip 02 — Pencarian hybrid (BM25 + vektor) dan reranker.

Membandingkan tiga cara: BM25 (kata kunci), vektor (kemiripan hashing), dan hybrid
(gabungan). Lalu menunjukkan reranker menyusun ulang hasil agar passage yang tepat
(berpola definisi "X adalah ...") berada di atas.

Jalankan:
    .venv\\Scripts\\python 04-rag-bps\\src\\02_retrieve.py
"""

from __future__ import annotations

from llmbepees_cl.rag import SAMPLE_DOCS, HybridRetriever, LexicalReranker, chunk_documents

passages = chunk_documents(SAMPLE_DOCS)
retriever = HybridRetriever(passages, alpha=0.5)

query = "Apa itu inflasi?"
print(f"query: {query!r}\n")
for rank, result in enumerate(retriever.search(query, top_k=3), start=1):
    print(f"{rank}. skor={result.score:.3f} (bm25={result.bm25:.2f}, vektor={result.vector:.2f})")
    print(f"   {result.passage.citation}: {result.passage.text[:70]}...")

print("\n--- Efek reranker ---")
q2 = "Jelaskan konsep sampel."
candidates = retriever.search(q2, top_k=5)
print(f"query: {q2!r}")
print("sebelum rerank (top-1):", candidates[0].passage.text[:70], "...")
ranked = LexicalReranker().rerank(q2, candidates, top_k=3)
print(
    f"sesudah rerank (top-1, match={ranked[0].match_score:.2f}):",
    ranked[0].passage.text[:70],
    "...",
)
