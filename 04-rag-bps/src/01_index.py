"""Skrip 01 — Membangun indeks: dokumen → passage ber-sitasi.

Alur RAG dimulai dari dokumen publik, dipecah menjadi passage kecil per bagian
logis, lalu diindeks agar bisa dicari. Di sini kita pakai korpus contoh bawaan
(`SAMPLE_DOCS`); dokumen publik Anda bisa ditaruh di `04-rag-bps/data/`.

Jalankan:
    .venv\\Scripts\\python 04-rag-bps\\src\\01_index.py
"""

from __future__ import annotations

from llmbepees_cl.rag import SAMPLE_DOCS, chunk_documents

passages = chunk_documents(SAMPLE_DOCS)
print(f"dokumen: {len(SAMPLE_DOCS)} | passage: {len(passages)}")
print("(korpus contoh, ditulis sendiri; dokumen publik nyata diletakkan di 04-rag-bps/data/)\n")

for passage in passages[:4]:
    print(f"- [{passage.passage_id}] {passage.citation} — {passage.section}")
    print(f"    {passage.text}")
print("...")
print("\nCatatan: setiap passage membawa judul, tahun, dan bagian sebagai bahan sitasi.")
