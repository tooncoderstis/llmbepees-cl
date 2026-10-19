# Catatan — Fase 5: Mekanika training, scaling, serving (M9–10)

Format tiap entri: **apa yang dicoba · hasil · apa yang dipelajari**.

## Ringkasan
- [x] Mekanika training: LR schedule (warmup+cosine), gradient accumulation, clipping, mixed precision (bf16 GPU-only)
- [x] Konsep training terdistribusi (DDP/FSDP/tensor-expert parallel) — dipelajari, praktik GPU menyusul
- [x] Scaling law kecil (sintetis, data tak terbatas) + pasang hukum pangkat
- [ ] llama.cpp/Ollama + kuantisasi (Q4/Q8) & KV cache; vLLM di GPU — konsep, belum dijalankan
- [x] Serve LLM lokal nyata (Hugging Face + adaptor LoRA) sebagai chat web FastAPI (`05_chat_llm.py`)
- [x] Bungkus asisten jadi API FastAPI + benchmark latensi

## Cara menjalankan
```sh
.venv\Scripts\python -m pip install -e ".[serve]"
.venv\Scripts\python 05-serving\01_train_mechanics.py
.venv\Scripts\python 05-serving\02_scaling.py
.venv\Scripts\python 05-serving\03_serve.py       # server; buka http://127.0.0.1:8000/docs
.venv\Scripts\python 05-serving\04_benchmark.py
$env:HF_HOME='E:\hf'; .venv\Scripts\python 05-serving\05_chat_llm.py   # chat; buka http://127.0.0.1:8000
```
Kode reusable: `src/llmbepees_cl/training/` + `src/llmbepees_cl/serving/` (diuji di `tests/`).

## Log
- **Mekanika training** — `train_with_accumulation`: micro-batch × akumulasi = batch efektif tanpa menambah memori puncak; clipping membatasi norma gradien; `autocast_context` bf16 otomatis no-op di CPU.
  Pelajaran: warmup mencegah langkah besar di awal; akumulasi adalah cara melatih "batch besar" di RAM terbatas.

- **01 · Demo mekanika (MiniGPT)** — jadwal LR 0.05 → 1.0 (warmup 20) → ~0 (step 200); latih dengan micro-batch 8 × akumulasi 4, clipping 1.0: loss **3.77 → 1.67**.
  Pelajaran: tiga alat ini cukup untuk membuat training kecil stabil dan efisien.

- **02 · Scaling law (sintetis)** — tugas `f(a,b)=(a·5+b·3) mod 12` (data tak terbatas). Params 7k/26k/102k → loss val 0.53/0.32/0.32; hukum pangkat `loss ≈ 2.66·N^(-0.19)`.
  Pelajaran: memakai korpus teks kecil **salah arah** (model besar overfit → loss val naik); scaling law butuh data & komputasi cukup. Tugas sintetis memisahkan efek kapasitas dari efek hafalan.

- **03 · API FastAPI** — `GET /health`, `POST /ask` (kembalikan answer/citations/tool_calls/abstained), `GET /indicators`; diuji dengan `TestClient` (konfigurasi kosong → 422).
  Pelajaran: kontrak `Answer` Fase 4 langsung jadi kontrak HTTP; jejak audit tetap.

- **04 · Benchmark latensi (in-process)** — 5 pertanyaan × 50 ulangan: rata-rata ~0,2 ms, p50 ~0,13 ms, p95 ~0,5 ms (ekstraktif, CPU).
  Pelajaran: ini baseline; LLM generatif nanti jauh lebih lambat → caching/KV-cache/kuantisasi jadi relevan.

- **05 · Chat lokal (LLM + LoRA)** — `05_chat_llm.py`: FastAPI `GET /` (halaman HTML) + `POST /chat`
  yang memuat `Qwen2.5-0.5B-Instruct` + adaptor LoRA **sekali** (CPU, fp32), lalu menjawab via
  chat template dengan riwayat. Terverifikasi lokal: `GET /` → 200 HTML, `POST /chat` → balasan
  model. Uji CI memakai **engine palsu** (`tests/test_chat_serving.py`) tanpa memuat model.
  Pelajaran: memuat model sekali (bukan per permintaan) adalah kunci UX; di CPU tiap balasan
  lambat (puluhan detik) sehingga caching/KV-cache/kuantisasi (llama.cpp/Ollama) jadi langkah
  berikutnya. Adaptor kita belum mengubah perilaku (0/3), jadi ini uji *pipeline*, bukan mutu.

## Latihan yang disarankan
1. Ganti mesin jawab ekstraktif dengan `LLMAnswerer` (Ollama/llama.cpp) dan bandingkan latensi vs kualitas.
2. Ubah fungsi sintetis scaling jadi butuh konteks lebih panjang; amati eksponen.
3. Ukur end-to-end latency lewat HTTP (uvicorn + permintaan paralel).

## Catatan jujur
- Mixed precision & distributed training tidak dijalankan di laptop (CPU/1 GPU) — di sini hanya konsep + kode yang siap dipakai di GPU.
- `/ask` masih diserve oleh asisten ekstraktif Fase 4; jalur LLM kini punya endpoint terpisah (`/chat`, `05_chat_llm.py`) — belum dijahit ke kontrak `Answer` (sitasi/tool/abstain).
