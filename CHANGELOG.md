# Changelog

Semua perubahan penting didokumentasikan di file ini.

Format mengikuti [Keep a Changelog](https://keepachangelog.com) dan proyek ini menganut [Semantic Versioning](https://semver.org).

Status terkini: [`STATUS.md`](STATUS.md).

## [Unreleased]
### Added
- Fase 5 (Mekanika training, scaling, serving): paket `src/llmbepees_cl/training/` (jadwal LR warmup+cosine, akumulasi gradien + clipping + autocast bf16, fit hukum pangkat) dan `src/llmbepees_cl/serving/` (API FastAPI: `/health`, `/ask`, `/indicators`; benchmark latensi p50/p95) + tes pytest.
- Skrip belajar `05-serving/` (01 mekanika training, 02 scaling law sintetis, 03 serve, 04 benchmark) + `PANDUAN.md` dan `notes.md`; ekstra `serve` ditambah `httpx`.
- Fase 4 (RAG + tool calling): paket `src/llmbepees_cl/rag/` — korpus contoh publik + pemuat dokumen, chunking ber-metadata, BM25 from-scratch, embedding hashing (+ opsi `sentence-transformers`), retrieval hybrid, reranker sadar-definisi, tool `get_indicator`, prompt sitasi/abstain, dan asisten ekstraktif ber-sitasi; + tes pytest.
- Skrip belajar `04-rag-bps/src/` (01 indeks → 04 tanya) dan evaluasi `04-rag-bps/eval/` (test set 52 pertanyaan + `run_eval.py`, metrik akurasi/sitasi/abstain/angka-via-tool); dilengkapi `PANDUAN.md` dan `notes.md`.
- Fase 3 (Fine-tuning LoRA): paket `src/llmbepees_cl/lora/` — `LoRALinear` from-scratch (+ `merged_weight`), `apply_lora` (membekukan bobot dasar), util hitung parameter, + tes pytest.
- Skrip belajar `03-finetune-lora/`: `01_lora_from_scratch.py`, `02_finetune_lora_minigpt.py` (CPU), serta skrip Colab/GPU `finetune_openweight_colab.py` + `eval_format_colab.py` untuk model open-weight (Qwen2.5-0.5B-Instruct); dilengkapi `PANDUAN.md`, `COLAB.md`, `notes.md`.
- Fase 2 (Transformer mini dari nol): paket `src/llmbepees_cl/minigpt/` — tokenizer karakter & BPE sederhana, self-attention kausal multi-head, blok transformer + `MiniGPT` (pre-norm), training prediksi token berikutnya, dan generasi teks (temperature/top-k) + tes pytest.
- Skrip belajar `02-mini-gpt/` (01 tokenizer → 05 generate) beserta `PANDUAN.md` dan `notes.md`; korpus kecil bawaan `TINY_CORPUS` (tanpa unduhan).
- Fase 1 (Dasar PyTorch): paket `src/llmbepees_cl/basics/` (data sintetis & MNIST opsional, `MLP`, training loop, cek gradient numerik) + tes pytest.
- Skrip belajar `01-pytorch-basics/` (01 NumPy → 06 MNIST opsional) beserta `notes.md`.
- Ekstra opsional `vision` (torchvision) di `pyproject.toml`; CI memasang torch CPU dari index CPU.
### Changed
- `requirements.txt`: `torch` diaktifkan (CPU); catatan pemasangan via index CPU.

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
