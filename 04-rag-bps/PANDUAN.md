# Panduan Belajar — Fase 4: RAG + tool calling (M7–8)

Tujuan: membangun prototipe **asisten BPS** yang (1) menjawab konsep/definisi dari
dokumen publik yang di-retrieve **dengan sitasi**, (2) menyajikan angka resmi
**hanya** dari tool, dan (3) **abstain** bila sumber tidak ada. Kaitannya ke
`ROADMAP.md`: "Minggu 7–8: RAG dan tool calling".

- **Kode reusable**: `src/llmbepees_cl/rag/` (diuji di `tests/test_rag_*.py`).
- **Korpus contoh**: `SAMPLE_DOCS` di paket (ditulis sendiri). Dokumen publik nyata → `04-rag-bps/data/`.
- **Evaluasi**: `04-rag-bps/eval/testset.jsonl` (52 pertanyaan) + `run_eval.py`.
- **Catatan hasil**: `notes.md`.
- Semua berjalan **CPU tanpa unduhan model** (embedding bawaan = hashing).

Pola tiap bagian: **Apa yang dilakukan · Kenapa begitu · Jalankan sendiri · Latihan.**

---

## Prinsip asisten (jangan dilanggar)

1. Konsep/definisi **hanya** dari konteks ter-retrieve, selalu **ber-sitasi** (judul, tahun, bagian).
2. Angka resmi **hanya** dari tool; sebutkan **periode, wilayah, satuan**.
3. **Tidak menebak**: bila konteks/tool tidak memuat → "informasi tidak ditemukan".

Ini yang membedakannya dari "LLM biasa": faktualitas dari **grounding**, bukan dari bobot model.

---

## Arsitektur pipeline

```
pertanyaan
  ├─ routing: konsep atau angka?
  ├─ angka  -> tool get_indicator(...)  -> angka + periode/wilayah/satuan + sumber
  └─ konsep -> chunking -> indeks -> retrieval hybrid (BM25+vektor) -> reranker
                 -> pilih passage terbaik -> jawaban + sitasi
  └─ bila skor/match rendah -> abstain "informasi tidak ditemukan"
```

---

## Skrip 01 — `src/01_index.py` (dokumen → passage)

**Apa yang dilakukan.** Memecah dokumen menjadi passage per bagian logis dan menampilkan
metadata sitasinya.

**Kenapa begitu.** LLM punya batas konteks dan pencarian lebih tajam pada unit kecil.
*Chunking* per gagasan utuh (satu definisi per passage) membuat retrieval presisi dan
sitasi jelas. Kita simpan `title`, `year`, `section` supaya jawaban bisa dirujuk.

**Jalankan sendiri.**
```sh
.venv\Scripts\python 04-rag-bps\src\01_index.py
```

**Latihan.** 1) Ubah `max_chars` di `chunk_documents` — kapan satu definisi terbelah? 2) Taruh file `.txt` publik di `04-rag-bps/data/` lalu muat dengan `load_text_files`.

---

## Skrip 02 — `src/02_retrieve.py` (retrieval hybrid + reranker)

**Apa yang dilakukan.** Membandingkan BM25 (kata kunci), vektor (hashing), hybrid, lalu
menunjukkan reranker menaikkan passage definisi yang tepat.

**Kenapa begitu.** BM25 unggul untuk istilah persis; vektor untuk sinonim/parafrase;
**hybrid** (weighted fusion) lebih tahan dari keduanya. **Reranker** menyempurnakan urutan:
di sini leksikal + sadar-definisi (bonus untuk pola "X adalah ..."), sehingga "Apa itu
populasi?" memilih passage "Populasi adalah ..." bukan passage yang sekadar menyebut kata itu.

**Jalankan sendiri.**
```sh
.venv\Scripts\python 04-rag-bps\src\02_retrieve.py
```
Perhatikan: skor hybrid `inflasi` = BM25 1.00 + vektor 0.14; rerank mengubah top-1 dari
"Definisi operasional ..." menjadi "Sampel adalah ...".

**Latihan.** 1) Ubah `alpha` (0 = BM25 saja, 1 = vektor saja) — kapan BM25 lebih baik?
2) Ganti `HashingEmbedder` dengan `SentenceTransformerEmbedder` (butuh unduhan) — bandingkan.

---

## Skrip 03 — `src/03_tools.py` (tool angka)

**Apa yang dilakukan.** Memanggil `get_indicator(...)`, melihat perilaku "tidak ada"
(`None`), dan mendeteksi indikator dari pertanyaan.

**Kenapa begitu.** Angka resmi tidak boleh "diingat" model. Tool adalah batas tegas:
ia mengembalikan nilai **beserta** wilayah, periode, satuan, dan sumber — inilah jejak
audit yang menjawab KPI anti-halusinasi. Data di sini **contoh (dummy)**; ganti dengan
sumber/API publik resmi saat tersedia.

**Jalankan sendiri.**
```sh
.venv\Scripts\python 04-rag-bps\src\03_tools.py
```

**Latihan.** 1) Tambah indikator baru (periode lain) di `tools.py`. 2) Buat tool `list_publications()` yang mengembalikan daftar dokumen publik.

---

## Skrip 04 — `src/04_ask.py` (prototipe asisten)

**Apa yang dilakukan.** Menjalankan pipeline lengkap dan menampilkan `Answer` terstruktur
(teks, sitasi, pemanggilan tool, status abstain).

**Kenapa begitu.** Ini "aturan main" asisten dalam kode: routing konsep vs angka, jawaban
ber-sitasi, dan abstain. Pada Fase 4 jawaban konsep **ekstraktif** (mengutip kalimat) agar
bisa diuji tanpa model generatif; Fase 5 mengganti mesin jawab dengan LLM tanpa mengubah
kontrak `Answer`.

**Jalankan sendiri.**
```sh
.venv\Scripts\python 04-rag-bps\src\04_ask.py
```

**Latihan.** 1) Tambah pertanyaan angka tanpa tahun — pastikan abstain. 2) Ubah `abstain_threshold` dan lihat efeknya ke penolakan.

---

## Evaluasi — `eval/run_eval.py` + `testset.jsonl`

**Apa yang dilakukan.** Menjalankan 52 pertanyaan (konsep, angka, di-luar-korpus) dan
menghitung akurasi, ketepatan sitasi, abstain benar, dan disiplin "angka via tool".

**Kenapa begitu.** PRD menuntut metrik eksplisit, bukan klaim anekdotal. Baseline ini
**in-domain dan disusun bersama korpus**, jadi angkanya optimistis — test set sesungguhnya
(200–500 pertanyaan, jawaban acuan dari ahli) menyusul di Fase 6.

**Jalankan sendiri.**
```sh
.venv\Scripts\python 04-rag-bps\eval\run_eval.py
```
Acuan (Fase 4): akurasi 100% (44/44), sitasi 100% (34/34), abstain 100% (8/8), angka via tool 100% (10/10).

**Latihan.** 1) Tambah pertanyaan "jebakan" (definisi mirip). 2) Turunkan `abstain_threshold` — berapa banyak abstain yang salah?

---

## Verifikasi

```sh
.venv\Scripts\ruff check .
.venv\Scripts\ruff format --check .
.venv\Scripts\pytest
```

Setelah Fase 4, lanjut ke **Fase 5 — Mekanika training, scaling, dan serving** (`05-serving/`):
bungkus asisten ini menjadi API FastAPI, ganti mesin jawab dengan LLM lokal, dan ukur latensi.
