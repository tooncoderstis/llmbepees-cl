# ADR-0004: Evaluasi via harness bersama + test set dibangun dari korpus (draf)

- **Status**: Diterima
- **Tanggal**: 2026-10-19

## Konteks
Fase 4 memerlukan metrik eksplisit (akurasi, sitasi, abstain, angka-via-tool) dan Fase 6
memerlukan benchmark yang lebih besar serta analisis kegagalan. Test set "ideal" (200–500
pertanyaan dengan jawaban acuan dari **ahli substansi BPS**) belum tersedia dan tidak bisa
dihasilkan otomatis secara kredibel. Sementara itu proyek harus terus teruji dan terukur di
tiap perubahan, di laptop (CPU), tanpa bergantung data internal.

## Keputusan
1. Satukan logika evaluasi dalam paket reusable `src/llmbepees_cl/evaluation/` (harness + metrik + analisis kegagalan), dipakai bersama oleh `04-rag-bps/eval/` dan `06-specialization/eval/`.
2. Test set besar dibangun **secara deterministik** dari korpus `SAMPLE_DOCS` dan baris tool (`build_testset.py`): metrik `must_contain` diambil dari teks definisi, sitasi dari metadata passage.
3. Status test set ini adalah **draf placeholder**; validasi/penyusunan jawaban acuan oleh ahli substansi tetap diperlukan (dan dicatat sebagai item terbuka di `STATUS.md`).

## Alasan
- DRY: satu harness menghindari menduplikasi logika metrik di dua folder.
- Deterministik & transparan: test set dapat direproduksi (`build_testset.py`) dan tak bergantung pada data internal.
- Menjaga DoD: setiap perubahan tetap punya metrik, walau test set final belum ada.
- Sesuai aturan data: hanya memakai korpus contoh publik yang ditulis sendiri.

## Konsekuensi
- **Positif**: evaluasi konsisten lintas fase; mudah menambah kategori uji; laporan JSON seragam.
- **Negatif**: angka tes in-domain cenderung **optimistis** (100%) dan tidak boleh dianggap bukti kualitas sesungguhnya.
- **Lain-lain**: saat test set ahli tersedia, ganti isi `testset.jsonl` (format sama) tanpa mengubah kode; jangan hapus test set draf (berguna sebagai regresi cepat).

## Alternatif yang dipertimbangkan
- **Menunggu test set ahli dulu** — ditolak; menghambat pengujian tiap fase.
- **Menyalin test set dari proyek lain** — ditolak; tidak relevan & berisiko lisensi/data.
- **Menduplikasi metrik per folder** — ditolak; rawan divergen.
