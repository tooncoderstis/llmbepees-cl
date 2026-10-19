# Catatan — Fase 4: RAG + tool calling (M7–8)

Format tiap entri: **apa yang dicoba · hasil · apa yang dipelajari**.

## Ringkasan
- [x] Korpus: contoh publik bawaan (`SAMPLE_DOCS`) + pemuat untuk `data/` (dokumen publik nyata)
- [x] Chunking per bagian logis, embedding (hashing; multibahasa opsional), hybrid BM25+vektor, reranker
- [x] Prompt sitasi wajib & perilaku "tidak ditemukan"
- [x] Tool `get_indicator(...)` (data dummy) dengan periode/wilayah/satuan/sumber
- [x] Test set awal 52 pertanyaan + `eval/run_eval.py`

## Cara menjalankan
```sh
.venv\Scripts\python 04-rag-bps\src\01_index.py
.venv\Scripts\python 04-rag-bps\src\02_retrieve.py
.venv\Scripts\python 04-rag-bps\src\03_tools.py
.venv\Scripts\python 04-rag-bps\src\04_ask.py
.venv\Scripts\python 04-rag-bps\eval\run_eval.py
```
Kode reusable: `src/llmbepees_cl/rag/` (diuji di `tests/test_rag_*.py`).

## Log
- **01 · Chunking & indeks** — 6 dokumen → **28 passage**; tiap passage membawa judul/tahun/bagian untuk sitasi.
  Pelajaran: chunk per gagasan (satu definisi per passage) membuat retrieval presisi; metadata = bahan sitasi.

- **02 · Retrieval hybrid + reranker** — query "Apa itu inflasi?": BM25 1.00, vektor 0.14, hybrid 0.568 (top-1 benar).
  Reranker mengubah top-1 "Jelaskan konsep sampel." dari *"Definisi operasional …"* menjadi *"Sampel adalah …"* (match=2.00).
  Pelajaran: hybrid menutup kelemahan masing-masing; **bonus definisi** (pola "X adalah …") penting karena BM25/vektor sering memilih passage yang hanya menyebut kata kunci.

- **03 · Tool angka** — `get_indicator("inflasi","Indonesia","2023")` → 3.67 persen (sumber: contoh dummy).
  Tanpa periode / indikator tak dikenal → `None` → abstain.
  Pelajaran: tool = batas tegas angka vs memori model; hasil selalu membawa periode/wilayah/satuan (jejak audit).

- **04 · Asisten** — konsep → jawaban ber-sitasi; angka → pemanggilan tool; di luar korpus / tanpa periode → "informasi tidak ditemukan".
  Pelajaran: kontrak `Answer` (teks, sitasi, tool_calls, abstained) membuat perilaku bisa diuji dan diaudit.

- **Evaluasi (baseline Fase 4)** — 52 pertanyaan: akurasi **100%** (44/44), sitasi **100%** (34/34),
  abstain **100%** (8/8), angka-via-tool **100%** (10/10). Semua di atas target KPI PRD.
  **Catatan jujur:** test set ini in-domain dan disusun bersama korpus, jadi angkanya optimistis.
  Rencana: perbaikan pertama muncul saat menambahkan pertanyaan "jebakan" & definisi mirip (mis. populasi vs sampel, periode vs satuan) — di situ reranker sempat salah dan kami tambahkan bonus definisi.

## Latihan yang disarankan
1. Tambah dokumen publik nyata di `04-rag-bps/data/` lalu ulang evaluasi dengan pertanyaan baru.
2. Ganti embedder ke `SentenceTransformerEmbedder` (multibahasa) — ukur dampaknya ke recall.
3. Tambah pertanyaan angka dengan periode berbeda; pastikan tidak ada angka yang keluar tanpa tool.

## Desain berikutnya (lihat `docs/adr/` bila jadi keputusan)
- Ganti mesin jawab ekstraktif → LLM lokal (Fase 5), kontrak `Answer` tetap.
- Biarkan `SentenceTransformerEmbedder` menjadi default bila model tersedia (unduh sekali, simpan di E:/F:).
