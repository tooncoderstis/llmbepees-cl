"""Menyusun prompt asisten: sitasi wajib, abstain, dan angka hanya dari tool.

Prompt ini dipakai saat nanti dihubungkan ke LLM (Fase 5). Pada Fase 4, pipeline
menjawab secara **ekstraktif** (mengambil kalimat dari dokumen) supaya prototipe
sudah bisa dijalankan dan diukur tanpa model generatif.
"""

from __future__ import annotations

from llmbepees_cl.rag.chunking import Passage

SYSTEM_PROMPT = (
    "Anda asisten statistik BPS. Aturan:\n"
    "1. Jawab konsep/definisi hanya dari KONTEKS yang diberikan, dan sertakan sitasi "
    "(judul, tahun, bagian).\n"
    "2. Angka resmi hanya boleh berasal dari hasil tool; sebutkan periode, wilayah, dan satuan.\n"
    "3. Jika jawaban tidak ada di konteks maupun tool, katakan: 'informasi tidak ditemukan'.\n"
    "4. Jangan menebak dan jangan mengarang angka."
)


def build_context(passages: list[Passage]) -> str:
    """Rangkai passage menjadi blok konteks ber-nomor dengan sitasi."""
    if not passages:
        return "(konteks kosong)"
    lines = []
    for i, passage in enumerate(passages, start=1):
        lines.append(f"[{i}] {passage.citation} — bagian {passage.section}: {passage.text}")
    return "\n".join(lines)


def format_tool_result(record: dict[str, object]) -> str:
    """Ubah hasil tool menjadi satu baris teks yang menyebut periode/wilayah/satuan."""
    return (
        f"{record['indicator']} {record['region']} periode {record['period']}: "
        f"{record['value']} {record['unit']} (sumber: {record['source']})"
    )


def build_prompt(
    question: str,
    passages: list[Passage],
    tool_results: list[dict[str, object]] | None = None,
) -> str:
    """Susun prompt lengkap (system + konteks + hasil tool + pertanyaan)."""
    tool_block = "(tidak ada)"
    if tool_results:
        tool_block = "\n".join(format_tool_result(r) for r in tool_results)
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"KONTEKS:\n{build_context(passages)}\n\n"
        f"HASIL TOOL:\n{tool_block}\n\n"
        f"PERTANYAAN: {question}\nJAWABAN:"
    )
