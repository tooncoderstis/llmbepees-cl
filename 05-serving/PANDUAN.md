# Panduan Belajar — Fase 5: Mekanika training, scaling, serving (M9–10)

Tiga topik: (1) **mekanika training** yang dipakai di lab (jadwal LR, akumulasi
gradien, clipping, mixed precision), (2) **scaling laws** (loss vs ukuran model),
(3) **serving** — membungkus asisten Fase 4 menjadi API + mengukur latensi.
Kaitannya ke `ROADMAP.md`: "Minggu 9–10".

- **Prasyarat serving**: `.venv\Scripts\python -m pip install -e ".[serve]"` (FastAPI, uvicorn, httpx).
- **Kode reusable**: `src/llmbepees_cl/training/` (diuji `tests/test_training_*.py`) dan `src/llmbepees_cl/serving/` (diuji `tests/test_serving.py`).
- **Catatan hasil**: `notes.md`.
- Training di laptop hanya untuk **konsep** (CPU, model mini). Latihan GPU/distribusi di Colab/sewaan.

Pola tiap bagian: **Apa yang dilakukan · Kenapa begitu · Jalankan sendiri · Latihan.**

---

## Konsep: mekanika training

- **Warmup + peluruhan kosinus.** LR naik linier di awal (hindari langkah besar saat gradien liar), lalu turun mengikuti kosinus agar mengendap stabil di akhir.
- **Akumulasi gradien.** Jalankan beberapa *micro-batch*, jumlahkan gradiennya, baru `step()` — *batch efektif* = micro-batch × jumlah akumulasi, tanpa menambah memori puncak. Ini kunci melatih di RAM/VRAM terbatas.
- **Gradient clipping.** Batasi norma gradien (mis. 1.0) supaya satu batch "buruk" tidak meledakkan bobot.
- **Mixed precision (bf16).** Berhitung di bfloat16 → hemat memori & cepat; **praktis hanya di GPU**. Di CPU kita pakai float32 (autocast dimatikan otomatis oleh `autocast_context`).
- **Terlatih terdistribusi** (konsep, praktik di GPU): **DDP** (replikasi model, bagi data), **FSDP** (pecah parameter/optimizer/gradien antar-GPU), **tensor/expert parallel** (pecah satu matriks / pakai MoE). Tujuannya: skala model/data melebihi satu GPU.

---

## Skrip 01 — `01_train_mechanics.py`

**Apa yang dilakukan.** Menampilkan kurva jadwal LR, lalu melatih MiniGPT dengan
akumulasi gradien (micro-batch 8 × 4 = batch efektif 32), warmup+cosine, dan clipping.

**Kenapa begitu.** Ini contoh "resep" training lab dalam skala kecil: tanpa akumulasi,
batch besar tidak muat; tanpa warmup, awal training bisa tidak stabil.

**Jalankan sendiri.**
```sh
.venv\Scripts\python 05-serving\01_train_mechanics.py
```
Contoh: LR 0.05 → 1.0 (warmup) → ~0 (akhir); loss 3.77 → ~1.67.

**Latihan.** 1) Ubah `accum_steps` jadi 1 dan 8 — apa efeknya ke loss? 2) Lepas `max_grad_norm` — apa yang terjadi saat satu batch besar? 3) Ubah `warmup_steps` jadi 0 (tanpa warmup).

---

## Skrip 02 — `02_scaling.py`

**Apa yang dilakukan.** Melatih MiniGPT dengan lebar berbeda (**data sintetis tak terbatas**:
token berikutnya = `(a·5 + b·3) mod 12`), mencatat loss validasi vs jumlah parameter, lalu
memasang hukum pangkat `L = a·N^b`.

**Kenapa begitu.** Scaling laws (Kaplan 2020; Chinchilla 2022) menyatakan loss menurun
sebagai hukum pangkat terhadap parameter/data/komputasi. Kita pakai tugas sintetis agar
loss validasi mengukur **generalisasi** — kalau memakai korpus teks kecil, model besar
**overfit** dan loss validasi justru naik (pelajaran: scaling butuh data & komputasi cukup).

**Jalankan sendiri.**
```sh
.venv\Scripts\python 05-serving\02_scaling.py
```
Contoh: params 7k/26k/102k → loss 0.53/0.32/0.32; eksponen `b ≈ -0.19` (negatif = makin besar makin baik).

**Latihan.** 1) Ganti fungsi sintetis jadi lebih kompleks (butuh ingatan lebih panjang) — apakah eksponen berubah? 2) Tambah titik lebar model; kapan tren mulai "datar" (saturasi)?

---

## Skrip 03 — `03_serve.py` (API)

**Apa yang dilakukan.** Menjalankan server FastAPI: `GET /health`, `POST /ask`, `GET /indicators`.

**Kenapa begitu.** Ini pembungkusan asisten Fase 4 jadi layanan yang bisa dipanggil
program lain. Kontrak `Answer` (teks, sitasi, tool_calls, abstained) tetap — jejak audit
terjaga. Dokumentasi otomatis di `/docs` (Swagger).

**Jalankan sendiri.**
```sh
.venv\Scripts\python 05-serving\03_serve.py
```
Lalu dari PowerShell lain:
```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
Invoke-RestMethod http://127.0.0.1:8000/ask -Method Post -ContentType "application/json" -Body '{"question":"Apa itu inflasi?"}'
```

**Latihan.** 1) Tambah endpoint `/retrieve` yang mengembalikan passage teratas. 2) Batasi panjang `question` untuk mencegah penyalahgunaan.

---

## Skrip 04 — `04_benchmark.py` (latensi)

**Apa yang dilakukan.** Mengukur latensi `ask()` in-process (rata-rata, p50, p95).

**Kenapa begitu.** Latensi adalah NFR (PRD). Baseline ekstraktif ini ~0,2 ms/pertanyaan;
saat mesin jawab diganti LLM (Fase 5 lanjut), angka ini jadi pembanding — LLM akan
jauh lebih lambat, sehingga caching/kv-cache & kuantisasi jadi penting.

**Jalankan sendiri.**
```sh
.venv\Scripts\python 05-serving\04_benchmark.py
```

**Latihan.** 1) Tambah pertanyaan panjang; bandingkan p95. 2) Ukur juga end-to-end via HTTP (TestClient/uvicorn).

---

## Skrip 05 — `05_chat_llm.py` (chat lokal dengan LLM)

**Apa yang dilakukan.** Menjalankan chat web lokal: `GET /` (halaman HTML) + `POST /chat`,
membungkus `Qwen2.5-0.5B-Instruct` + adaptor LoRA (`src/llmbepees_cl/serving/chat.py`).

**Kenapa begitu.** Ini langkah "serve LLM lokal" yang sesungguhnya: model dimuat **sekali**
saat balasan pertama (lazy), riwayat percakapan dijaga di browser → server **stateless**.
`transformers`/`peft` diimpor di dalam engine, jadi CI tetap bisa menguji endpoint dengan
**engine palsu** (tanpa mengunduh model).

**Jalankan sendiri.**
```sh
$env:HF_HOME='E:\hf'
.venv\Scripts\python 05-serving\05_chat_llm.py
# buka http://127.0.0.1:8000
```
Atur lewat env: `CHAT_ADAPTER_DIR=none` (base saja), `CHAT_MAX_NEW_TOKENS`, `CHAT_MODEL_ID`.

**Latihan.** 1) Ganti adaptor ke hasil DPO; bandingkan. 2) Tambah **streaming** (token demi token) dengan `TextIteratorStreamer`. 3) Ukur latensi `/chat` (p50/p95) seperti Skrip 04.

---

## Konsep serving (praktik di GPU/Colab)

- **llama.cpp / Ollama**: menjalankan model kecil lokal (CPU pun bisa), dengan **kuantisasi** (Q4/Q8) yang mengecilkan memori dengan sedikit penurunan mutu.
- **KV cache**: menyimpan key/value attention yang sudah dihitung agar token berikutnya tidak dihitung ulang — kunci kecepatan inferensi generasi.
- **vLLM**: server inferensi ber-throughput tinggi (paged attention) — coba di GPU sewaan.
- Di sini kita **sudah** menyerve LLM lokal (Skrip 05, Hugging Face + LoRA di CPU); llama.cpp/Ollama + kuantisasi menjadi langkah lanjut untuk kecepatan. Kontrak `Answer` tetap siap untuk dijahit ke jalur LLM.

---

## Verifikasi

```sh
.venv\Scripts\ruff check .
.venv\Scripts\ruff format --check .
.venv\Scripts\pytest
```

Setelah Fase 5, lanjut ke **Fase 6 — Jalur lab model: post-training & evaluasi** (`06-specialization/`):
preference tuning (DPO), dasar RL for LLM, dan benchmark evaluasi sendiri.
