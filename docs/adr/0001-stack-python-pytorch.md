# ADR-0001: Stack Python (NumPy → PyTorch CPU), tooling pip+venv+ruff+pytest

- **Status**: Diterima
- **Tanggal**: 2026-10-17

## Konteks
Proyek ini adalah jurnal belajar pengembangan model AI (roadmap 12 minggu) dan prototipe asisten BPS.
Pemilik adalah pengguna Python yang terbiasa mengolah data. Lingkungan kerja bertumpu di laptop
Windows 11 dengan RAM 8 GB dan GPU MX230 2 GB (tanpa CUDA), jadi training berat tidak realistis di sini.
Proyek serupa `llmbepees` sudah memakai Python 3.11 + NumPy/PyTorch + pip/venv/ruff/pytest, sehingga
sudah ada acuan gaya pengerjaan yang terbukti.

## Keputusan
Bahasa utama **Python 3.11**. Eksperimen dimulai dari **NumPy** sebagai rujukan kebenaran numerik,
lalu **PyTorch versi CPU** untuk training. Tooling: **pip + virtualenv (`.venv`)** untuk dependensi,
**ruff** untuk lint/format, **pytest** untuk test. Node hanya dipakai untuk skrip kecil (pelaporan hub).

## Alasan
- Python + NumPy/PyTorch adalah jalur standar belajar dan riset LLM; ekosistem evaluasi & tokenizer paling lengkap.
- Konsisten dengan `llmbepees` → konvensi, perintah, dan pola uji bisa dipakai ulang (hemat waktu belajar peralatan).
- NumPy-first membuat konsep (matmul, softmax, backprop) terlihat eksplisit sebelum dibungkus framework.
- pip+venv paling familiar dan tanpa alat baru; ruff cepat dan menyatukan lint+format.
- CPU-only memaksa efisiensi (model kecil, konteks pendek) yang justru baik untuk pembelajaran.

## Konsekuensi
- **Positif**: konsep terlihat transparan; alur kerja familiar; mudah dijalankan di laptop.
- **Negatif**: eksperimen skala besar terbatas; perlu pindah ke Colab/Kaggle/GPU sewaan untuk training Lanjut.
- **Lain-lain**: `torch` sebagai dependensi opsional (extra) agar instalasi dasar ringan; disiplin menjaga RAM.

## Alternatif yang dipertimbangkan
- **uv (resolusi lebih cepat)** — ditunda; pip+venv sudah cukup dan konsisten dengan `llmbepees`.
- **Hugging Face `transformers`+`datasets` sebagai basis** — dipakai sebagai referensi nanti, tapi tidak untuk
  fondasi belajar (ingin melihat mekanismenya dulu).
- **Node/JS atau Rust** — ekosistem LLM-riset kalah matang untuk tujuan pembelajaran ini.
