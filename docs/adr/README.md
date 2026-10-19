# Architecture Decision Records (ADR)

Catatan keputusan teknis penting beserta konteks dan konsekuensinya.
Format mengikuti [ADR gaya Nygard](https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions).

Cara menambah ADR:
1. Salin `0000-template.md` → `NNNN-judul-singkat.md` (nomor urut berikutnya).
2. Isi konteks, keputusan, alasan, konsekuensi, alternatif.
3. Tambahkan baris ke tabel di bawah.

Status: **Diterima** · **Ditolak** · **Digantikan oleh ADR-XXXX** · **Deprecated**.

| ADR | Judul | Status | Tanggal |
|---|---|---|---|
| [0001](0001-stack-python-pytorch.md) | Stack Python (NumPy → PyTorch CPU), tooling pip+venv+ruff+pytest | Diterima | 2026-10-17 |
| [0002](0002-deploy-easypanel.md) | Deploy EasyPanel via Docker image (programatik MCP/CLI) | Diterima | 2026-10-17 |
| [0003](0003-proyek-terpisah-dari-llmbepees.md) | Proyek terpisah dari `llmbepees` | Diterima | 2026-10-17 |
