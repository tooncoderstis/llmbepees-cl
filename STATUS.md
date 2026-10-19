# STATUS — llmbepees-cl

> **Sumber kebenaran status & progres.** Baca file ini dulu sebelum melanjutkan pekerjaan.
> Riwayat rilis: [`CHANGELOG.md`](CHANGELOG.md) · Keputusan: [`docs/adr/`](docs/adr/) · Kebutuhan: [`PRD.md`](PRD.md).
> **Jangan simpan kredensial di file ini.**

- **Terakhir diperbarui**: 2026-10-20
- **Versi produksi aktif**: belum ada (sekali deploy baru di Fase 5)
- **Platform**: Python 3.11 (NumPy → PyTorch CPU); ruff + pytest; hub portofolio via Node

## TL;DR (konteks 30 detik)
`llmbepees-cl` adalah **jurnal belajar 12 minggu** (`ROADMAP.md`) yang sekaligus membangun **prototipe asisten BPS faktual & ber-sitasi** (konsep/definisi via RAG, angka hanya via tool). Ini **proyek baru terpisah** dari `llmbepees` (ADR-0003). Stack Python 3.11, PyTorch CPU, pip+venv+ruff+pytest. **Fase 0–2 selesai** — scaffold, dasar PyTorch (`01-pytorch-basics/`), dan transformer mini dari nol (`02-mini-gpt/`) dengan paket ber-tes `basics/` + `minigpt/`. **Fase 4 (RAG + tool calling) selesai** — paket `rag/` + skrip `04-rag-bps/` & evaluasi (baseline 100%, in-domain). **Fase 5 (mekanika training, scaling, serving) selesai** — paket `training/` & `serving/` (API FastAPI + benchmark latensi) + skrip `05-serving/`. **Fase 6 (post-training & evaluasi) selesai** — paket `posttraining/` (DPO from-scratch) & `evaluation/` (harness + analisis kegagalan) + skrip `06-specialization/` (DPO CPU, benchmark **200 pertanyaan** 100%, ringkasan paper) + ADR-0004. **Fase 3 (LoRA) dikerjakan** — paket `lora/` + skrip CPU `03-finetune-lora/` ber-tes; **Colab T4 sudah dijalankan** (Qwen2.5-0.5B-Instruct, 0,436% parameter, loss 3.91→2.71) namun eval format masih 0/3 — perlu lebih banyak data/langkah. Repo publik `tooncoderstis/llmbepees-cl`, CI hijau. Berikutnya: susun test set ahli (ADR-0004); tambah data/langkah Fase 3/6 agar perilaku berubah.

## Checklist Tahap
| Fase | Judul | Status |
|---|---|---|
| 0 | Setup & Fondasi (scaffold repo, tooling, CI, hub) | ✅ |
| 1 | Dasar PyTorch (M1–2) | ✅ |
| 2 | Transformer mini dari nol (M3–4) | ✅ |
| 3 | Fine-tuning LoRA (M5–6) | 🔄 |
| 4 | RAG + tool calling: prototipe asisten BPS (M7–8) | ✅ |
| 5 | Mekanika training, scaling, serving (M9–10) | ✅ |
| 6 | Jalur lab model: post-training & evaluasi (M11–12) | ✅ |

> Fase dikerjakan: tandai 🔄 saat mulai dan ✅ saat selesai, lalu jalankan `npm run hub:report` di tiap perubahan — hub portofolio mengirim notifikasi (inbox + Web Push).

## Cara melanjutkan (resume) — baca ini saat membuka proyek lagi
1. **Baca `STATUS.md` ini** — kondisi & langkah berikutnya ada di sini.
2. **Pasang lingkungan**: `python -m venv .venv` lalu `.venv\Scripts\python -m pip install -e ".[dev]"`.
3. **Verifikasi cepat**: `.venv\Scripts\ruff check .` dan `.venv\Scripts\pytest`.
4. **Setelah 12 minggu** (`ROADMAP.md`): jalankan Colab Fase 3 & DPO open-weight (GPU), lalu langkah lanjut — reproduksi paper kecil, kontribusi open-source, tulisan teknis, pilih bidang (infrastruktur training / evals-data / alignment).

## Kondisi saat ini
- Scaffold awal dibuat dari skill `aasaprojectkit`: dokumen (PRD/STATUS/CHANGELOG/ADR), health files, CI, scaffold deploy, dan hub.
- Fase 1 selesai: `torch 2.14.1+cpu` + `torchvision 0.29.1+cpu` terpasang di `.venv` (CUDA off, 4 thread); CI juga memasang torch CPU.
- Skrip belajar di `01-pytorch-basics/` (01–06) + paket reusable `src/llmbepees_cl/basics/` ber-tes. MNIST subset → akurasi uji 0.889.
- Fase 2 selesai: `02-mini-gpt/` (01 tokenizer → 05 generate) + paket `src/llmbepees_cl/minigpt/` ber-tes. Model `n_embd=64, n_head=4, n_layer=2, block_size=64` (109k parameter) pada `TINY_CORPUS`: loss latih 3.77 → 0.18 (validasi 4.45 — overfit data kecil); generasi menghasilkan teks statistik yang masuk akal.
- Fase 3 (berjalan): paket `src/llmbepees_cl/lora/` + skrip `03-finetune-lora/` (01 LoRA from-scratch, 02 LoRA pada MiniGPT, skrip Colab). CPU: LoRA ~3.6% parameter dilatih, loss contoh tahanan turun (2.26 → 1.98). **Colab T4 sudah dijalankan (2026-10-20)**: Qwen2.5-0.5B-Instruct, `trainable% 0.436%`, loss 3.913 → 2.705 (10 langkah); eval format dasar 0/3 vs LoRA 0/3 (jawaban identik — belum cukup data/langkah). Adaptor disimpan di Drive (`MyDrive/llm-bps/`). Temuan kompatibilitas Colab: uninstall `torchao`, `SFTConfig.max_length`, T4 → fp16 (bukan bf16). Adaptor LoRA & DPO diunduh ke lokal (gitignored); **inferensi lokal CPU jalan** lewat `03-finetune-lora/03_ask_local.py` (adaptor aktif, jawaban masih identik — belum terlatih cukup; base model di `HF_HOME=E:\hf`).
- Fase 4 selesai: paket `src/llmbepees_cl/rag/` + skrip `04-rag-bps/` (01 indeks → 04 tanya) & evaluasi `04-rag-bps/eval/` (52 pertanyaan). Baseline: akurasi 100% (44/44), sitasi 100% (34/34), abstain 100% (8/8), angka-via-tool 100% (10/10) — in-domain, jadi optimistis. Angka hanya dari tool `get_indicator` (data dummy); konsep ber-sitasi dari korpus contoh.
- Fase 5 selesai: paket `src/llmbepees_cl/training/` + `src/llmbepees_cl/serving/` + skrip `05-serving/` (01 mekanika → 04 benchmark). Baseline latensi asisten (ekstraktif, in-process): p50 ~0,13 ms / p95 ~0,5 ms. Scaling law sintetis: loss vs parameter `~N^-0.19`. FastAPI diuji via `TestClient`. Mixed precision & distributed hanya konsep (CPU/1 GPU). **Chat LLM lokal** (Hugging Face + adaptor LoRA, CPU) jalan lewat `05-serving/05_chat_llm.py` (`GET /` halaman chat, `POST /chat`; diuji `tests/test_chat_serving.py` dengan engine palsu).
- Fase 6 selesai: paket `src/llmbepees_cl/posttraining/` (DPO from-scratch) + `src/llmbepees_cl/evaluation/` (harness) + skrip `06-specialization/`. DPO MiniGPT (CPU): loss `ln 2 → ~0.003`, akurasi preferensi `0 → 1.0`, reward dipilih/ditolak `(0,0) → (2.5,-3.4)`. Benchmark **200 pertanyaan** (konsep/angka/abstain): semua metrik 100% (in-domain, optimistis). Ringkasan 6 paper di `papers/`. **Colab T4 sudah dijalankan (2026-10-20)**: QLoRA 4-bit, loss ln2 → 0.37 (5 langkah); adaptor di Drive. Temuan Colab: uninstall `torchao`, `processing_class`, T4 → fp16 & AMP off (bf16 tak didukung).
- Git lokal diinisialisasi; repo ter-push ke GitHub (`tooncoderstis/llmbepees-cl`, publik) — CI Actions hijau (ruff + 74 tes + gitleaks).

## Yang belum selesai / menunggu
| Item | Catatan |
|---|---|
| Repo GitHub | ✅ `tooncoderstis/llmbepees-cl` (publik) — 7 commit ter-push |
| Secret hub | ✅ `HUB_INGEST_SECRET` terkonfigurasi di `.env` (tidak di-commit); `hub:report` berjalan |
| CI | ✅ GitHub Actions hijau (ruff + 74 tes + gitleaks) |
| Deploy EasyPanel | ⬜ provision saat Fase 5 (asisten FastAPI belum ada) |
| Data dokumen BPS | ⬜ dokumen publik nyata belum diletakkan di `04-rag-bps/data/` (tes pakai korpus contoh `SAMPLE_DOCS`) |
| Fine-tune Colab (Fase 3) | ✅ dijalankan (T4, loss 3.91→2.71, 0,436% param) — hasil dicatat di `03-finetune-lora/notes.md`; perlu data/langkah lebih untuk ubah perilaku |
| Perbaiki skrip Colab (Fase 3) | ✅ patch permanen: `pick_dtype` (T4→fp16), `SFTConfig.max_length`, `processing_class`, uninstall `torchao` (+ `COLAB.md`) — terverifikasi di T4 |
| DPO Colab (Fase 6) | ✅ dijalankan (T4, QLoRA, loss ln2→0.37); catatan di `06-specialization/notes.md` |
| Test set evaluasi nyata | ⬜ ganti draf test set (dari korpus) dengan 200–500 pertanyaan + jawaban acuan ahli substansi (ADR-0004) |
| Serve LLM lokal (Fase 5) | 🔄 chat LLM lokal (HF + LoRA) jalan lewat `05-serving/05_chat_llm.py`; ⬜ jahit ke `/ask` (sitasi/tool/abstain) & kecepatan (Ollama/llama.cpp) |
| Latihan distributed/GPU (Fase 5) | ⬜ DDP/FSDP/bf16 praktik di Colab/sewaan (di laptop hanya konsep) |

## Cara menjalankan & menguji
```sh
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"
.venv\Scripts\ruff check .
.venv\Scripts\ruff format --check .
.venv\Scripts\pytest
npm run hub:report
```

## Rilis
`powershell -File deploy/easypanel-docker/build-push.ps1 -Image <registry>/llmbepees-cl -Tag <versi> -Push` → deploy di EasyPanel (programatik). Detail: [`docs/runbooks/deploy.md`](docs/runbooks/deploy.md).

## Referensi cepat
- PRD: `PRD.md` · Changelog: `CHANGELOG.md` · ADR: `docs/adr/README.md` · Runbook: `docs/runbooks/`
