# Changelog

Semua perubahan penting didokumentasikan di file ini.

Format mengikuti [Keep a Changelog](https://keepachangelog.com) dan proyek ini menganut [Semantic Versioning](https://semver.org).

Status terkini: [`STATUS.md`](STATUS.md).

## [Unreleased]
### Added
- Chat web lokal dengan LLM: `src/llmbepees_cl/serving/chat.py` (FastAPI; `GET /` halaman HTML + `POST /chat`; `LocalLLMEngine` memuat model + adaptor LoRA sekali, `transformers`/`peft` diimpor lazily) + skrip `05-serving/05_chat_llm.py` + tes `tests/test_chat_serving.py` (engine palsu, tanpa unduh model). Terverifikasi lokal di CPU.
- Inferensi lokal CPU: `03-finetune-lora/03_ask_local.py` (memuat base `Qwen2.5-0.5B-Instruct` + adaptor LoRA, membandingkan jawaban dasar vs LoRA) + extra `finetune` di `pyproject.toml` (`transformers`, `peft`, `datasets`, `accelerate`). Terverifikasi jalan di laptop (adaptor aktif, `lora_B absmean ≈ 5.9e-4`).
- Fase 6 (Post-training & evaluasi): paket `src/llmbepees_cl/posttraining/` (DPO from-scratch: `sequence_logprobs`, `dpo_loss`, `train_dpo`, `evaluate_dpo` + dataset preferensi) dan `src/llmbepees_cl/evaluation/` (harness metrik + analisis kegagalan) + tes pytest.
- Skrip `06-specialization/`: `01_dpo_from_scratch.py` (CPU), `02_dpo_openweight_colab.py` + `COLAB.md` (DPO/QLoRA Qwen2.5-0.5B-Instruct via TRL), `03_rl_concepts.md`; benchmark `eval/build_testset.py` (200 pertanyaan) + `eval/run_eval.py`; ringkasan paper `papers/README.md`; `PANDUAN.md` dan `notes.md`.
- ADR-0004: evaluasi via harness bersama + test set dibangun dari korpus (draf).
- Fase 5 (Mekanika training, scaling, serving): paket `src/llmbepees_cl/training/` (jadwal LR warmup+cosine, akumulasi gradien + clipping + autocast bf16, fit hukum pangkat) dan `src/llmbepees_cl/serving/` (API FastAPI: `/health`, `/ask`, `/indicators`; benchmark latensi p50/p95) + tes pytest.
- Skrip belajar `05-serving/` (01 mekanika training, 02 scaling law sintetis, 03 serve, 04 benchmark) + `PANDUAN.md` dan `notes.md`; ekstra `serve` ditambah `httpx`.
### Changed
- `eval_format_colab.py`: `device_map="auto"` hanya dipakai saat CUDA tersedia — di CPU ini membuat bobot adaptor nyangkut di `meta` (adaptor tak benar-benar terpasang).
- Skrip Colab disamakan dengan API/lingkungan Colab terbaru sehingga **jalan tanpa patch**:
  `pick_dtype()` memilih dtype dari compute capability (≥8 Ampere+ → bf16, T4 → **fp16**, CPU → fp32),
  `SFTConfig(max_length=...)` (bukan `max_seq_length`), `DPOTrainer(processing_class=tokenizer)`,
  AMP dimatikan di T4, plus catatan `!pip uninstall -y torchao` di docstring & `COLAB.md`.
  Terverifikasi berjalan di Colab T4 (`finetune`+`eval` Fase 3; DPO Fase 6).
- Fase 6 dijalankan di **Google Colab (Tesla T4)**: `02_dpo_openweight_colab.py` — QLoRA 4-bit +
  LoRA (r=16, β=0.1) pada `Qwen2.5-0.5B-Instruct` → loss **ln 2 (0.6931) → 0.3655** (5 langkah).
  Adaptor disimpan di Drive; temuan kompatibilitas (uninstall `torchao`, `processing_class`,
  T4 tanpa bf16 → fp16 & AMP off) dicatat di `06-specialization/notes.md`.
- Fase 3 dijalankan di **Google Colab (Tesla T4)**: `Qwen2.5-0.5B-Instruct` + LoRA (r=16, q/k/v/o)
  → 2.162.688/496.195.456 parameter dilatih (**0.436%**), loss 3.913 → 2.705 (10 langkah).
  Eval format: dasar 0/3, LoRA 0/3 (jawaban identik — belum terlatih cukup). Hasil & temuan
  kompatibilitas (torchao, `SFTConfig.max_length`, T4 tanpa bf16 → fp16) dicatat di
  `03-finetune-lora/notes.md`; adaptor disimpan di Drive.
- `04-rag-bps/eval/run_eval.py` memakai paket `evaluation` (DRY); `README.md` diperluas (peta fase, contoh, hasil ringkas); `dev` extra menambah `fastapi`+`httpx`.
- CI: `secret-scan` memakai gitleaks CLI (scan riwayat penuh) alih-alih action (gagal pada push pertama).
- `requirements.txt`: `torch` diaktifkan (CPU); catatan pemasangan via index CPU.
- Fase 4 (RAG + tool calling): paket `src/llmbepees_cl/rag/` — korpus contoh publik + pemuat dokumen, chunking ber-metadata, BM25 from-scratch, embedding hashing (+ opsi `sentence-transformers`), retrieval hybrid, reranker sadar-definisi, tool `get_indicator`, prompt sitasi/abstain, dan asisten ekstraktif ber-sitasi; + tes pytest.
- Skrip belajar `04-rag-bps/src/` (01 indeks → 04 tanya) dan evaluasi `04-rag-bps/eval/` (test set 52 pertanyaan + `run_eval.py`, metrik akurasi/sitasi/abstain/angka-via-tool); dilengkapi `PANDUAN.md` dan `notes.md`.
- Fase 3 (Fine-tuning LoRA): paket `src/llmbepees_cl/lora/` — `LoRALinear` from-scratch (+ `merged_weight`), `apply_lora` (membekukan bobot dasar), util hitung parameter, + tes pytest.
- Skrip belajar `03-finetune-lora/`: `01_lora_from_scratch.py`, `02_finetune_lora_minigpt.py` (CPU), serta skrip Colab/GPU `finetune_openweight_colab.py` + `eval_format_colab.py` untuk model open-weight (Qwen2.5-0.5B-Instruct); dilengkapi `PANDUAN.md`, `COLAB.md`, `notes.md`.
- Fase 2 (Transformer mini dari nol): paket `src/llmbepees_cl/minigpt/` — tokenizer karakter & BPE sederhana, self-attention kausal multi-head, blok transformer + `MiniGPT` (pre-norm), training prediksi token berikutnya, dan generasi teks (temperature/top-k) + tes pytest.
- Skrip belajar `02-mini-gpt/` (01 tokenizer → 05 generate) beserta `PANDUAN.md` dan `notes.md`; korpus kecil bawaan `TINY_CORPUS` (tanpa unduhan).
- Fase 1 (Dasar PyTorch): paket `src/llmbepees_cl/basics/` (data sintetis & MNIST opsional, `MLP`, training loop, cek gradient numerik) + tes pytest.
- Skrip belajar `01-pytorch-basics/` (01 NumPy → 06 MNIST opsional) beserta `notes.md`.
- Ekstra opsional `vision` (torchvision) di `pyproject.toml`; CI memasang torch CPU dari index CPU.

## [0.1.0] - 2026-10-17
### Added
- Scaffold proyek dari skill `aasaprojectkit`: `PRD.md`, `STATUS.md`, `CHANGELOG.md`.
- Dokumen: `docs/README.md`, `docs/adr/` (template + ADR-0001/0002/0003), `docs/runbooks/deploy.md`.
- Pembelajaran agen: seksi *Project Bootstrap* di `AGENTS.md` + `opencode.json`.
- Repo: `.gitignore`, `.gitattributes`, `.env.example`, `pyproject.toml`, `requirements.txt`, `package.json`.
- Health files: `README.md`, `CONTRIBUTING.md`, `SECURITY.md`, PR/issue templates, `CODEOWNERS`, `dependabot.yml`.
- CI GitHub Actions: `ruff` + `pytest` + gitleaks.
- Scaffold deploy EasyPanel (Docker image) + runbook.
- Pelaporan hub portofolio: `scripts/hub-report.mjs` + skrip npm `hub:report`.
