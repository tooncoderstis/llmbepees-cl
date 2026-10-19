# llmbepees-cl

Jurnal belajar 12 minggu membangun LLM dari awal (PyTorch, mini-GPT, LoRA, RAG, serving,
post-training & evaluasi) **sekaligus** prototipe **asisten BPS** yang faktual dan ber-sitasi:
konsep/definisi dijawab dari dokumen yang di-retrieve, angka resmi hanya dari output tool.

## Fitur
- Asisten tanya-jawab konsep/definisi statistik dengan **sitasi** (judul, tahun, bagian).
- Tool data resmi: angka hanya dari output tool, menyertakan periode/wilayah/satuan.
- **Abstain** ("tidak ditemukan") saat sumber tidak ada — bukan menebak.
- LoRA / DPO untuk menyesuaikan **perilaku** (format, sitasi), bukan menanamkan data.
- Catatan belajar & eksperimen transparan per fase (`notes.md`, `PANDUAN.md`).

## Stack
- Python 3.11, NumPy → PyTorch (CPU). Tooling: pip + venv + ruff + pytest.
- Serving: FastAPI (`.[serve]`). Deploy EasyPanel (Docker image, programatik).

## Peta fase
| Fase | Folder | Isi |
|---|---|---|
| 0 | — | Scaffold, tooling, CI, hub |
| 1 | `01-pytorch-basics/` | Tensor, autograd, training loop, MLP/MNIST |
| 2 | `02-mini-gpt/` | Tokenizer, self-attention, MiniGPT, generasi |
| 3 | `03-finetune-lora/` | LoRA from-scratch (CPU) + Colab model open-weight |
| 4 | `04-rag-bps/` | Chunking, BM25+vektor+reranker, tool, asisten ber-sitasi, eval |
| 5 | `05-serving/` | Mekanika training, scaling law, API FastAPI, benchmark |
| 6 | `06-specialization/` | DPO from-scratch, konsep RL, benchmark 200 pertanyaan, ringkasan paper |

Kode reusable ada di `src/llmbepees_cl/` (`basics`, `minigpt`, `lora`, `rag`, `training`, `serving`, `posttraining`, `evaluation`).

## Menjalankan
```sh
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"          # + ".[serve]" untuk API FastAPI
```
Contoh cepat:
```sh
.venv\Scripts\python 04-rag-bps\src\04_ask.py            # prototipe asisten (ber-sitasi)
.venv\Scripts\python 04-rag-bps\eval\run_eval.py         # evaluasi 52 pertanyaan
.venv\Scripts\python 06-specialization\01_dpo_from_scratch.py   # DPO dari nol (CPU)
.venv\Scripts\python 06-specialization\eval\run_eval.py  # benchmark 200 pertanyaan
.venv\Scripts\python 05-serving\03_serve.py              # API http://127.0.0.1:8000/docs
```

## Testing
```sh
.venv\Scripts\ruff check .
.venv\Scripts\ruff format --check .
.venv\Scripts\pytest
```

## Hasil ringkas (baseline, in-domain — lihat catatan kejujuran di tiap `notes.md`)
- RAG (52 & 200 pertanyaan): akurasi, sitasi, abstain, angka-via-tool **100%** (optimistis; test set diturunkan dari korpus yang sama, jawaban acuan ahli menyusul).
- DPO (MiniGPT, CPU): loss `ln 2 → ~0.003`, akurasi preferensi `0 → 1.0`.
- Latensi asisten ekstraktif (in-process): p50 ~0,13 ms / p95 ~0,5 ms.
- MiniGPT: loss latih 3.77 → 0.18 pada `TINY_CORPUS`.

## Dokumentasi
- Kebutuhan: [`PRD.md`](PRD.md)
- Status: [`STATUS.md`](STATUS.md)
- Rencana belajar: [`ROADMAP.md`](ROADMAP.md)
- Riwayat rilis: [`CHANGELOG.md`](CHANGELOG.md)
- Keputusan: [`docs/adr/`](docs/adr/)
- Runbook: [`docs/runbooks/`](docs/runbooks/)

## Kontribusi
Lihat [`CONTRIBUTING.md`](CONTRIBUTING.md).
