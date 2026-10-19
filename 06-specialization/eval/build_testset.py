"""Bangun test set evaluasi (~200 pertanyaan) secara deterministik dari korpus & tool.

Sumber pertanyaan:
- **konsep**: tiap definisi di `SAMPLE_DOCS` -> 5 variasi pertanyaan (must_contain berupa
  potongan definisi, sitasi = judul (tahun));
- **angka**: tiap baris tool `get_indicator` -> 2 variasi (nilai + satuan wajib muncul);
- **abstain**: daftar pertanyaan di luar korpus (topik umum yang tidak ada di korpus).

Hasil: `testset.jsonl` (satu JSON per baris). Definisi acuan (must_contain) otomatis dari
teks korpus, jadi **draf** — jawaban acuan final perlu validasi ahli substansi.

Jalankan:
    .venv\\Scripts\\python 06-specialization\\eval\\build_testset.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from llmbepees_cl.rag import SAMPLE_DOCS, all_records, chunk_documents

OUTPUT = Path(__file__).parent / "testset.jsonl"

CONCEPT_VARIANTS = [
    "Apa itu {term}?",
    "Apa definisi {term}?",
    "Jelaskan konsep {term}.",
    "Apa yang dimaksud dengan {term}?",
    "Definisi {term} adalah apa?",
]

# Pertanyaan di luar korpus (topik umum; sengaja tanpa kata kunci korpus).
ABSTAIN_QUESTIONS = [
    "Apa warna langit saat senja?",
    "Bagaimana cara membuat rendang?",
    "Siapa penemu lampu pijar?",
    "Apa makanan khas Jepang?",
    "Apa itu kucing persia?",
    "Bagaimana cara merawat tanaman hias?",
    "Apa ibu kota Prancis?",
    "Siapa pelatih tim nasional Brasil?",
    "Apa lagu kebangsaan Islandia?",
    "Bagaimana cuaca di London hari ini?",
    "Apa warna bendera BPS?",
    "Apa bahan utama kue lapis?",
    "Siapa pemenang piala dunia sepak bola 2014?",
    "Bagaimana cara menyetel gitar?",
    "Apa itu fotosintesis?",
    "Apa planet terbesar di tata surya?",
    "Bagaimana cara belajar bermain piano?",
    "Apa hewan tercepat di darat?",
    "Siapa presiden pertama Amerika Serikat?",
    "Apa gunung tertinggi di dunia?",
    "Bagaimana membuat kopi susu?",
    "Apa warna bulu panda?",
    "Siapa penulis novel Laskar Pelangi?",
    "Bagaimana cara mengendarai sepeda?",
    "Apa itu gravitasi?",
    "Apa bahasa resmi Mesir?",
    "Siapa komponis Simfoni Nomor Sembilan?",
    "Bagaimana cara menanam anggrek?",
    "Apa bahan pembuat kaca?",
    "Apa itu pelangi?",
    "Bagaimana cara membuat tempe?",
    "Siapa pelari tercepat di dunia?",
    "Apa makanan khas Italia?",
    "Bagaimana cara memainkan gitar akustik?",
    "Apa itu gunung berapi?",
    "Siapa sekretaris jenderal PBB saat ini?",
    "Apa warna buah mangga matang?",
    "Bagaimana cara membuat origami burung?",
    "Apa itu bintang jatuh?",
    "Bagaimana cara memelihara ikan hias?",
]


def concept_items() -> list[dict]:
    items = []
    for passage in chunk_documents(SAMPLE_DOCS):
        match = re.match(r"(.+?)\s+adalah\s+(.+)", passage.text.strip(), re.IGNORECASE)
        if not match:
            continue
        term = match.group(1).strip()
        definition = match.group(2).strip()
        words = definition.split()
        key = (" ".join(words[:4]) if len(words) >= 4 else definition).rstrip(".,")
        for variant in CONCEPT_VARIANTS:
            items.append(
                {
                    "id": f"c{len(items):03d}",
                    "type": "concept",
                    "question": variant.format(term=term),
                    "expect_citation": passage.citation,
                    "must_contain": [key],
                }
            )
    return items


def number_items() -> list[dict]:
    items = []
    for row in all_records():
        indicator = str(row["indicator"])
        region = str(row["region"])
        period = str(row["period"])
        unit_word = str(row["unit"]).split()[0]
        for template in ("Berapa {ind} {reg} {per}?", "Berapa nilai {ind} {reg} {per}?"):
            items.append(
                {
                    "id": f"n{len(items):03d}",
                    "type": "number",
                    "question": template.format(ind=indicator, reg=region, per=period),
                    "expect_tool": indicator,
                    "must_contain": [str(row["value"]), unit_word],
                }
            )
    return items


def abstain_items() -> list[dict]:
    return [
        {"id": f"a{i:03d}", "type": "abstain", "question": question, "should_abstain": True}
        for i, question in enumerate(ABSTAIN_QUESTIONS)
    ]


def main() -> None:
    items = concept_items() + number_items() + abstain_items()
    OUTPUT.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in items), encoding="utf-8"
    )
    counts: dict[str, int] = {}
    for item in items:
        counts[item["type"]] = counts.get(item["type"], 0) + 1
    print(f"test set: {len(items)} pertanyaan -> {OUTPUT}")
    print("per jenis:", counts)


if __name__ == "__main__":
    main()
