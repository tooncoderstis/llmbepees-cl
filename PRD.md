# PRD: llmbepees-cl

**Status**: draft — setup & fondasi (Fase 0) dikerjakan.
**Versi dokumen**: 0.1
**Terkait**: `STATUS.md`, `CHANGELOG.md`, `docs/adr/`

> Dokumen ini menjawab **apa & kenapa**. Progres ada di `STATUS.md`, riwayat rilis di `CHANGELOG.md`, keputusan teknis di `docs/adr/`.

## 1. Ringkasan Produk
`llmbepees-cl` adalah **jurnal belajar 12 minggu** untuk naik dari dasar PyTorch sampai memahami alur lab model (pretraining, SFT/fine-tuning, post-training, dan evaluasi) **sekaligus** membangun **prototipe asisten BPS**: asisten yang andal soal konsep/definisi statistik dan penyajian data resmi.

Masalah yang dijawab: asisten berbasis LLM mudah berhalusinasi pada angka dan definisi. Solusinya dua lapis — **konsep/definisi** dijawab dari dokumen yang di-retrieve (RAG) dengan sitasi (judul, tahun, bagian/halaman), sedangkan **angka/data resmi** hanya keluar dari output tool/API, tidak pernah dari ingatan model. Bila sumber tidak ditemukan, asisten menyatakan tidak ditemukan.

Ini **proyek belajar terpisah** dari `llmbepees` (lihat [ADR-0003](docs/adr/0003-proyek-terpisah-dari-llmbepees.md)): fokusnya adalah kurikulum `ROADMAP.md` yang dikerjakan bertahap, satu langkah kecil demi satu, dengan setiap eksperimen didokumentasikan.

## 2. Prinsip Desain Utama
- **Grounding dulu, bukan hafalan bobot.** Faktualitas berasal dari dokumen yang dikutip, bukan dari memori model.
- **Angka hanya dari tool.** Tidak ada angka resmi yang diklaim dari ingatan model; selalu sebut periode, wilayah, satuan, dan sumber referensi resmi.
- **Abstain lebih baik daripada mengarang.** Jika konteks/tool tidak memuat jawaban → katakan tidak ditemukan.
- **Belajar bertahap & transparan.** Satu langkah kecil → jalankan → verifikasi → lanjut. Setiap eksperimen punya `notes.md`.
- **Hanya data publik.** Prototipe tidak memakai data internal/sensitif (aturan instansi, lihat `AGENTS.md`).
- **CPU-first di laptop, GPU di Colab/sewaan.** Tidak ada training berat di laptop ini.

## 3. Peran Pengguna (Roles)
| Role | Deskripsi | Akses |
|---|---|---|
| Pembelajar (pemilik) | Mengikuti roadmap, menjalankan & mencatat eksperimen | Seluruh repo & eksperimen lokal |
| Ahli substansi BPS | Menyusun jawaban acuan test set evaluasi | Test set (200–500 pertanyaan) |
| Pengguna asisten | Tanya konsep/definisi & minta data resmi | API/UI (rate-limited), publik setelah deploy |
| Operator | Melatih, mendeploy, memantau | MCP/CLI EasyPanel, `.env` lokal |

## 4. Fase / Modul Pengembangan
```
FASE 0 → Setup & Fondasi (scaffold repo, tooling, CI) — dikerjakan
FASE 1 → Dasar PyTorch (M1–2) — 01-pytorch-basics/
FASE 2 → Transformer mini dari nol (M3–4) — 02-mini-gpt/
FASE 3 → Fine-tuning LoRA (M5–6) — 03-finetune-lora/
FASE 4 → RAG + tool calling: prototipe asisten BPS (M7–8) — 04-rag-bps/
FASE 5 → Mekanika training, scaling, serving (M9–10) — 05-serving/
FASE 6 → Jalur lab model: post-training & evaluasi (M11–12) — 06-specialization/
BACKLOG → DPO lanjutan, kuantisasi, multimodal, kontribusi open-source
```

### 4.4 RAG + tool calling (asisten BPS)
- **Deskripsi**: indeks dokumen publik (chunking per bagian logis, embedding multibahasa, pencarian hybrid vektor+BM25, reranker), prompt sitasi wajib, tool `get_indicator(...)` untuk data resmi.
- **Acceptance criteria**: jawaban konsep mengutip sumber (judul/tahun/bagian); pertanyaan definisi di luar korpus → menolak dengan benar; angka hanya muncul via tool dan menyertakan periode/wilayah/satuan.
- **Non-goals**: mengarang jawaban saat sumber tidak ada; menyimpan angka di bobot model.

### 4.5 Evaluasi (& serving)
- **Deskripsi**: test set berlabel (jawaban acuan dari ahli substansi) + skrip evaluasi.
- **Acceptance criteria**: akurasi, ketepatan sitasi, tingkat halusinasi angka, dan penolakan yang benar terukur; asisten dibungkus API (FastAPI) dengan latensi terukur.

## 5. Model Data (ERD)
Bukan aplikasi ber-basis data. Artefak yang dikelola: **dokumen publik** (`04-rag-bps/data/`), **indeks KB**, **tokenizer**, **checkpoint model**, **test set evaluasi** (`04-rag-bps/eval/`), **catatan eksperimen** (`notes.md`).

## 6. Non-Functional Requirements
- **Akurasi terukur**: metrik eksplisit pada test set, bukan klaim anekdotal.
- **Jejak audit**: setiap jawaban konsep punya sitasi; jawaban angka punya sumber tool + periode.
- **Keamanan**: `.env` di-ignore; secret scan di CI; tidak ada dokumen internal/privat atau kredensial di repo.
- **Portabilitas**: jalan di laptop (CPU, RAM 8 GB, konteks ≤ 4096); eksperimen GPU hanya di Colab/Kaggle/sewaan.

## 7. KPI & Metrik Keberhasilan
| Area | Metrik | Target |
|---|---|---|
| Faktualitas | Akurasi pada test set | ≥ 80% |
| Sitasi | Ketepatan sitasi (sumber benar) | ≥ 90% |
| Anti-halusinasi | Angka tanpa dasar tool | ≤ 5% |
| Penolakan | Benar abstain saat sumber tak ada | ≥ 90% |
| Retrieval | Recall@k relevan | ≥ 85% |
| Operasional (Fase 5) | `/health` layanan API | 200 |

## 8. Kepatuhan & Keamanan
- Hanya **dokumen publik** (publikasi, glosarium, metadata terbuka); data internal/tidak dipublikasikan **tidak** dipakai.
- Tidak ada kredensial/token/API key di repo; gunakan `.env` (di-`.gitignore`) dan `.env.example`.
- Bila ragu apakah data boleh dipakai → anggap tidak boleh dan tanyakan.
- Rujukan umum: OWASP Top 10/ASVS; UU PDP No. 27/2022 sebagai rambu privasi.

## 9. Risiko & Mitigasi
| # | Risiko | Dampak | Mitigasi |
|---|---|---|---|
| 1 | Keterbatasan laptop (RAM 8 GB, tanpa NVIDIA) | Tinggi | Model kecil, CPU-only, batch kecil, konteks ≤4096; training berat di Colab/sewaan |
| 2 | Halusinasi angka/definisi | Tinggi | Grounding + tool wajib; abstain; temperature rendah; evaluasi halusinasi |
| 3 | Data internal/dokumen sensitif bocor ke repo | Tinggi | Aturan data di `AGENTS.md`, `.gitignore`, secret scan, hanya dokumen publik |
| 4 | Lingkup belajar melebar (12 minggu molor) | Sedang | Fase kecil & terukur; satu langkah sekali; catat di `notes.md` |
| 5 | Retrieval meleset (recall rendah) | Sedang | Hybrid embedding+BM25, chunking & metadata, reranker |

## 10. Out of Scope
- Training skala produksi (butuh GPU + data besar).
- Fine-tuning untuk "menanamkan data/angka" (hanya untuk perilaku/format/sitasi).
- Multimodal, MoE, infrastruktur training terdistribusi (BACKLOG).
- Penggunaan data internal/sensitif BPS.

## 11. Ringkasan Keputusan
| # | Pertanyaan | Keputusan |
|---|---|---|
| 1 | Tujuan | Jurnal belajar 12 minggu + prototipe asisten BPS faktual & ber-sitasi |
| 2 | Hubungan ke `llmbepees` | Proyek baru terpisah (ADR-0003) |
| 3 | Bahasa/stack | Python 3.11, NumPy→PyTorch (CPU), ADR-0001 |
| 4 | Tooling | pip + venv + ruff + pytest, ADR-0001 |
| 5 | Deploy | EasyPanel Docker image (programatik), provision saat Fase 5, ADR-0002 |
