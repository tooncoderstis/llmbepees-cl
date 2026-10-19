"""Skrip 03 — Tool `get_indicator`: satu-satunya sumber angka.

Mendemonstrasikan pemanggilan tool untuk data (contoh/dummy), perilaku "tidak ada"
saat data tak tersedia, dan deteksi indikator dari pertanyaan. Ingat prinsip:
angka resmi hanya dari tool, selalu dengan periode, wilayah, dan satuan.

Jalankan:
    .venv\\Scripts\\python 04-rag-bps\\src\\03_tools.py
"""

from __future__ import annotations

from llmbepees_cl.rag import available_indicators, get_indicator
from llmbepees_cl.rag.tools import detect_indicator

print("indikator tersedia:", available_indicators(), "\n")

for indicator, period in [("inflasi", "2023"), ("inflasi", "1900"), ("garis kemiskinan", "2023")]:
    record = get_indicator(indicator, "Indonesia", period)
    if record is None:
        print(f"{indicator} {period}: -> tidak ditemukan (None) -> asisten akan abstain")
    else:
        print(
            f"{indicator} {period}: {record['value']} {record['unit']} @ {record['region']} "
            f"(sumber: {record['source']})"
        )

print()
for question in ["Berapa TPT Indonesia 2024?", "Apa itu inflasi?", "Berapa curah hujan 2023?"]:
    print(f"deteksi indikator {question!r} -> {detect_indicator(question)}")
