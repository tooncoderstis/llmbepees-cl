"""Analisis kegagalan: mengelompokkan jawaban yang salah agar bisa diperbaiki.

Setelah evaluasi, kita tidak sekadar tahu *berapa* yang gagal, tetapi *jenis* apa yang
gagal (konsep salah ambil passage, sitasi meleset, abstain keliru, angka tak lewat tool).
"""

from __future__ import annotations


def summarize_failures(failures: list[dict]) -> dict[str, list[dict]]:
    """Kelompokkan kegagalan berdasarkan `type` (concept/number/abstain)."""
    grouped: dict[str, list[dict]] = {}
    for failure in failures:
        grouped.setdefault(failure.get("type", "unknown"), []).append(failure)
    return grouped


def format_summary(grouped: dict[str, list[dict]]) -> str:
    """Ringkasan teks: jumlah & contoh kegagalan per jenis."""
    if not grouped:
        return "Tidak ada kegagalan."
    lines = []
    for kind, items in sorted(grouped.items()):
        lines.append(f"- {kind}: {len(items)} gagal")
        for item in items[:3]:
            lines.append(f"    [{item['id']}] {item['question']} -> {item['answer'][:70]}")
    return "\n".join(lines)
