"""Skrip 04 — Prototipe asisten: ajukan pertanyaan, lihat jawaban ber-sitasi.

Menjalankan pipeline lengkap: routing (konsep vs angka) → retrieval + tool →
jawaban ber-sitasi atau abstain. Pada Fase 4 jawaban konsep masih **ekstraktif**
(mengutip kalimat dari dokumen); penggantian ke LLM dilakukan di Fase 5.

Jalankan:
    .venv\\Scripts\\python 04-rag-bps\\src\\04_ask.py
"""

from __future__ import annotations

from llmbepees_cl.rag import build_default_assistant

assistant = build_default_assistant()

questions = [
    "Apa itu indeks harga konsumen?",  # konsep → sitasi
    "Jelaskan konsep sampel.",  # konsep → sitasi
    "Berapa inflasi Indonesia 2023?",  # angka → tool (periode+wilayah+satuan)
    "Berapa jumlah penduduk Indonesia 2024?",  # angka → tool
    "Apa warna bendera BPS?",  # di luar korpus → abstain
    "Berapa inflasi Indonesia?",  # angka tanpa periode → abstain
]

for question in questions:
    answer = assistant.ask(question)
    tag = "ABSTAIN" if answer.abstained else "JAWAB"
    print(f"[{tag}] {question}")
    print(f"    {answer.text}")
    if answer.tool_calls:
        print(f"    tool: {answer.tool_calls[0]}")
    print()
