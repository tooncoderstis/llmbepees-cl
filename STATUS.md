# STATUS — llmbepees-cl

> **Sumber kebenaran status & progres.** Baca file ini dulu sebelum melanjutkan pekerjaan.
> Riwayat rilis: [`CHANGELOG.md`](CHANGELOG.md) · Keputusan: [`docs/adr/`](docs/adr/) · Kebutuhan: [`PRD.md`](PRD.md).
> **Jangan simpan kredensial di file ini.**

- **Terakhir diperbarui**: 2026-10-19
- **Versi produksi aktif**: belum ada (sekali deploy baru di Fase 5)
- **Platform**: Python 3.11 (NumPy → PyTorch CPU); ruff + pytest; hub portofolio via Node

## TL;DR (konteks 30 detik)
`llmbepees-cl` adalah **jurnal belajar 12 minggu** (`ROADMAP.md`) yang sekaligus membangun **prototipe asisten BPS faktual & ber-sitasi** (konsep/definisi via RAG, angka hanya via tool). Ini **proyek baru terpisah** dari `llmbepees` (ADR-0003). Stack Python 3.11, PyTorch CPU, pip+venv+ruff+pytest. **Fase 0–2 selesai** — scaffold, dasar PyTorch (`01-pytorch-basics/`), dan transformer mini dari nol (`02-mini-gpt/`) dengan paket ber-tes `basics/` + `minigpt/`. **Fase 3 (Fine-tuning LoRA, M5–6) dikerjakan** — paket `src/llmbepees_cl/lora/` + skrip CPU `03-finetune-lora/` ber-tes; skrip Colab/GPU (`finetune_openweight_colab.py`) siap namun **belum dijalankan**. **Fase 4 (RAG + tool calling, M7–8) selesai** — paket `src/llmbepees_cl/rag/` (chunking, BM25+hashing, hybrid+reranker, tool `get_indicator`, asisten ekstraktif ber-sitasi) + skrip `04-rag-bps/` & evaluasi 52 pertanyaan (baseline akurasi/sitasi/abstain/angka-via-tool 100%). **Fase 5 (Mekanika training, scaling, serving, M9–10) selesai** — paket `src/llmbepees_cl/training/` (jadwal LR, akumulasi gradien+clipping, bf16, scaling law) & `src/llmbepees_cl/serving/` (API FastAPI `/health` `/ask` `/indicators` + benchmark latensi) + skrip `05-serving/`. Berikutnya: Fase 6 (post-training & evaluasi); selesaikan run Colab Fase 3.

## Checklist Tahap
| Fase | Judul | Status |
|---|---|---|
| 0 | Setup & Fondasi (scaffold repo, tooling, CI, hub) | ✅ |
| 1 | Dasar PyTorch (M1–2) | ✅ |
| 2 | Transformer mini dari nol (M3–4) | ✅ |
| 3 | Fine-tuning LoRA (M5–6) | 🔄 |
| 4 | RAG + tool calling: prototipe asisten BPS (M7–8) | ✅ |
| 5 | Mekanika training, scaling, serving (M9–10) | ✅ |
| 6 | Jalur lab model: post-training & evaluasi (M11–12) | ⬜ |

> Fase dikerjakan: tandai 🔄 saat mulai dan ✅ saat selesai, lalu jalankan `npm run hub:report` di tiap perubahan — hub portofolio mengirim notifikasi (inbox + Web Push).

## Cara melanjutkan (resume) — baca ini saat membuka proyek lagi
1. **Baca `STATUS.md` ini** — kondisi & langkah berikutnya ada di sini.
2. **Pasang lingkungan**: `python -m venv .venv` lalu `.venv\Scripts\python -m pip install -e ".[dev]"`.
3. **Verifikasi cepat**: `.venv\Scripts\ruff check .` dan `.venv\Scripts\pytest`.
4. **Lanjutkan Fase 6 (post-training & evaluasi)**: jalur lab model — preference tuning (DPO), dasar RL for LLM, benchmark evaluasi sendiri di `06-specialization/`. Opsional: ganti mesin jawab `rag` dengan LLM lokal (Ollama/llama.cpp) lalu ukur latensi lagi. Fase 3 (run Colab) tetap menunggu.

## Kondisi saat ini
- Scaffold awal dibuat dari skill `aasaprojectkit`: dokumen (PRD/STATUS/CHANGELOG/ADR), health files, CI, scaffold deploy, dan hub.
- Fase 1 selesai: `torch 2.14.1+cpu` + `torchvision 0.29.1+cpu` terpasang di `.venv` (CUDA off, 4 thread); CI juga memasang torch CPU.
- Skrip belajar di `01-pytorch-basics/` (01–06) + paket reusable `src/llmbepees_cl/basics/` ber-tes. MNIST subset → akurasi uji 0.889.
- Fase 2 selesai: `02-mini-gpt/` (01 tokenizer → 05 generate) + paket `src/llmbepees_cl/minigpt/` ber-tes. Model `n_embd=64, n_head=4, n_layer=2, block_size=64` (109k parameter) pada `TINY_CORPUS`: loss latih 3.77 → 0.18 (validasi 4.45 — overfit data kecil); generasi menghasilkan teks statistik yang masuk akal.
- Fase 3 (berjalan): paket `src/llmbepees_cl/lora/` + skrip `03-finetune-lora/` (01 LoRA from-scratch, 02 LoRA pada MiniGPT, + skrip Colab). CPU: LoRA ~3.6% parameter dilatih, loss contoh tahanan turun (2.26 → 1.98). Skrip Colab/GPU siap namun belum dijalankan.
- Fase 4 selesai: paket `src/llmbepees_cl/rag/` + skrip `04-rag-bps/` (01 indeks → 04 tanya) & evaluasi `04-rag-bps/eval/` (52 pertanyaan). Baseline: akurasi 100% (44/44), sitasi 100% (34/34), abstain 100% (8/8), angka-via-tool 100% (10/10) — in-domain, jadi optimistis. Angka hanya dari tool `get_indicator` (data dummy); konsep ber-sitasi dari korpus contoh.
- Fase 5 selesai: paket `src/llmbepees_cl/training/` + `src/llmbepees_cl/serving/` + skrip `05-serving/` (01 mekanika → 04 benchmark). Baseline latensi asisten (ekstraktif, in-process): p50 ~0,13 ms / p95 ~0,5 ms. Scaling law sintetis: loss vs parameter `~N^-0.19`. FastAPI diuji via `TestClient`. Mixed precision & distributed hanya konsep (CPU/1 GPU).
- Git lokal diinisialisasi; repo ter-push ke GitHub (`tooncoderstis/llmbepees-cl`, publik) — CI Actions hijau (ruff + 74 tes + gitleaks).

## Yang belum selesai / menunggu
| Item | Catatan |
|---|---|
| Repo GitHub | ✅ `tooncoderstis/llmbepees-cl` (publik) — 7 commit ter-push |
| Secret hub | ✅ `HUB_INGEST_SECRET` terkonfigurasi di `.env` (tidak di-commit); `hub:report` berjalan |
| CI | ✅ GitHub Actions hijau (ruff + 74 tes + gitleaks) |
| Deploy EasyPanel | ⬜ provision saat Fase 5 (asisten FastAPI belum ada) |
| Data dokumen BPS | ⬜ dokumen publik nyata belum diletakkan di `04-rag-bps/data/` (tes pakai korpus contoh `SAMPLE_DOCS`) |
| Test set evaluasi nyata | ⬜ test set 200–500 pertanyaan + jawaban acuan ahli substansi (Fase 6); baseline 52 pertanyaan sudah ada |
| Fine-tune Colab (Fase 3) | ⬜ jalankan `03-finetune-lora/finetune_openweight_colab.py` di GPU; catat hasil di `notes.md` |
| Serve LLM lokal (Fase 5) | ⬜ ganti mesin jawab ekstraktif `rag` dengan LLM lokal (Ollama/llama.cpp) lalu ukur latensi |
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
