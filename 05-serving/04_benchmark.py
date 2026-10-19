"""Skrip 04 — Benchmark latensi asisten (in-process, tanpa server).

Mengukur waktu jawab `assistant.ask(...)` untuk beberapa pertanyaan, berulang kali,
lalu melaporkan rata-rata, p50, dan p95 (ms). Ini baseline sebelum mengganti mesin
jawab ekstraktif dengan LLM: LLM biasanya jauh lebih lambat, jadi angka ini jadi
titik banding.

Jalankan:
    .venv\\Scripts\\python 05-serving\\04_benchmark.py
"""

from __future__ import annotations

from llmbepees_cl.rag import build_default_assistant
from llmbepees_cl.serving import benchmark_ask

assistant = build_default_assistant()
questions = [
    "Apa itu inflasi?",
    "Jelaskan konsep sampel.",
    "Berapa inflasi Indonesia 2023?",
    "Berapa jumlah penduduk Indonesia 2024?",
    "Apa warna bendera BPS?",
]

result = benchmark_ask(assistant, questions, repeats=50)
print(f"pertanyaan: {len(questions)} | sampel: {result.n}")
print(f"rata-rata: {result.mean_ms:.2f} ms")
print(f"p50      : {result.p50_ms:.2f} ms")
print(f"p95      : {result.p95_ms:.2f} ms")
print(f"min/max  : {result.min_ms:.2f} / {result.max_ms:.2f} ms")
