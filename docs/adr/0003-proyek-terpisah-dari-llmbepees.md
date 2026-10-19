# ADR-0003: Proyek terpisah dari `llmbepees`

- **Status**: Diterima
- **Tanggal**: 2026-10-17

## Konteks
Sudah ada proyek `E:\aasatech\llmbepees` yang matang: starter dari `aasallm`, Python 3.11 (NumPy→PyTorch),
FastAPI, CI, dan deploy EasyPanel. Folder baru `llmbepeescl` lahir dari brainstorming terpisah (bersama
Claude Desktop) berupa `ROADMAP.md` 12 minggu + `AGENTS.md`, dan pada Fase 0 hampir kosong.
Muncul pertanyaan: apakah ini kelanjutan/rebranding `llmbepees`, atau proyek baru.

## Keputusan
`llmbepees-cl` adalah **proyek baru terpisah** dengan repo, STATUS, dan entri hub **sendiri**
(slug hub `llmbepees-cl`). Ia **tidak** menimpa atau menggantikan `llmbepees`. Konvensi pengerjaan
(pip/venv/ruff/pytest, `package.json` untuk hub, STATUS ber-TL;DR) diadopsi ulang dari kebiasaan Aasa.

## Alasan
- Tujuan berbeda: `llmbepees` fokus produk asisten faktual; `llmbepees-cl` fokus **kurikulum belajar 12 minggu**
  (dari dasar PyTorch → mini-GPT → LoRA → RAG → scaling → post-training) yang bertahap dan terdokumentasi.
- Memisahkan repo menjaga STATUS/CHANGELOG tiap proyek tetap fokus dan tidak saling mengaburkan.
- Hub portofolio butuh slug unik; `llmbepees-cl` jelas menandai varian kurikulum.

## Konsekuensi
- **Positif**: progres belajar terlihat mandiri; tidak mengganggu produk `llmbepees`.
- **Negatif**: ada duplikasi konvensi/config kecil (bisa disalin, bukan masalah).
- **Lain-lain**: jika kelak hasil terbaik dari sini ingin dipromosikan, dapat dipindahkan ke `llmbepees` secara eksplisit.

## Alternatif yang dipertimbangkan
- **Gabung ke `llmbepees` (mono-repo)** — ditolak; mengaburkan status produk vs kurikulum dan menambah beban.
- **Rebranding `llmbepees` → `llmbepees-cl`** — ditolak; merusak sejarah & deploy yang sudah berjalan.
