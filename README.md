# llmbepees-cl

Jurnal belajar 12 minggu membangun LLM dari awal (PyTorch, mini-GPT, LoRA, RAG, serving,
post-training & evaluasi) **sekaligus** prototipe **asisten BPS** yang faktual dan ber-sitasi:
konsep/definisi dijawab dari dokumen yang di-retrieve, angka resmi hanya dari output tool.

## Fitur (target akhir)
- Asisten tanya-jawab konsep/definisi statistik dengan **sitasi** (judul, tahun, bagian).
- Tool data resmi: angka hanya dari output tool, menyertakan periode/wilayah/satuan.
- **Abstain** ("tidak ditemukan") saat sumber tidak ada — bukan menebak.
- Catatan belajar & eksperimen transparan per fase (`notes.md`).

## Stack
- Python 3.11, NumPy → PyTorch (CPU). Tooling: pip + venv + ruff + pytest.
- Deploy (Fase 5): FastAPI via EasyPanel (Docker image, programatik).

## Menjalankan
```sh
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"
```

## Testing
```sh
.venv\Scripts\ruff check .
.venv\Scripts\pytest
```

## Dokumentasi
- Kebutuhan: [`PRD.md`](PRD.md)
- Status: [`STATUS.md`](STATUS.md)
- Rencana belajar: [`ROADMAP.md`](ROADMAP.md)
- Riwayat rilis: [`CHANGELOG.md`](CHANGELOG.md)
- Keputusan: [`docs/adr/`](docs/adr/)
- Runbook: [`docs/runbooks/`](docs/runbooks/)

## Kontribusi
Lihat [`CONTRIBUTING.md`](CONTRIBUTING.md).
