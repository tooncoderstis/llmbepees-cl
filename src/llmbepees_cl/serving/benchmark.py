"""Benchmark latensi asisten tanpa menjalankan server (in-process).

Mengukur waktu `assistant.ask(...)` untuk sekumpulan pertanyaan berulang, lalu
melaporkan rata-rata, median (p50), dan p95. Berguna untuk memantau latensi
sebelum/sesudah mengganti mesin jawab dengan LLM (Fase 5).
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass

from llmbepees_cl.rag import BPSAssistant, build_default_assistant


@dataclass
class BenchResult:
    n: int
    mean_ms: float
    p50_ms: float
    p95_ms: float
    min_ms: float
    max_ms: float


def percentile(values: list[float], pct: float) -> float:
    """Persentil linear (interpolasi) tanpa dependensi eksternal."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (pct / 100.0) * (len(ordered) - 1)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    frac = rank - low
    return ordered[low] * (1 - frac) + ordered[high] * frac


def benchmark_ask(
    assistant: BPSAssistant,
    questions: list[str],
    *,
    repeats: int = 5,
    warmup: int = 1,
) -> BenchResult:
    """Ukur latensi `assistant.ask` untuk tiap pertanyaan, diulang `repeats` kali."""
    for question in questions:
        for _ in range(warmup):
            assistant.ask(question)

    samples: list[float] = []
    for _ in range(repeats):
        for question in questions:
            start = time.perf_counter()
            assistant.ask(question)
            samples.append((time.perf_counter() - start) * 1000.0)

    return BenchResult(
        n=len(samples),
        mean_ms=statistics.fmean(samples),
        p50_ms=percentile(samples, 50),
        p95_ms=percentile(samples, 95),
        min_ms=min(samples),
        max_ms=max(samples),
    )


def default_questions() -> list[str]:
    """Pertanyaan contoh untuk benchmark (konsep, angka, abstain)."""
    return [
        "Apa itu inflasi?",
        "Apa itu indeks harga konsumen?",
        "Berapa inflasi Indonesia 2023?",
        "Berapa jumlah penduduk Indonesia 2024?",
        "Apa warna bendera BPS?",
    ]


def main() -> None:
    result = benchmark_ask(build_default_assistant(), default_questions(), repeats=20)
    print(f"sampel: {result.n}")
    print(
        f"rata-rata: {result.mean_ms:.2f} ms | p50: {result.p50_ms:.2f} ms | "
        f"p95: {result.p95_ms:.2f} ms"
    )
    print(f"min: {result.min_ms:.2f} ms | max: {result.max_ms:.2f} ms")


if __name__ == "__main__":
    main()
