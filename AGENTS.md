# AGENTS.md — Proyek Belajar AI + Asisten BPS

## Project Bootstrap (WAJIB DIBACA DULU)

> Hemat token & jaga konteks antar sesi. Urutan baca:
> 1. `STATUS.md` — kondisi & pekerjaan terkini (satu-satunya sumber kebenaran progres).
> 2. `PRD.md` — **jangan baca penuh**; hanya buka section yang relevan dengan tugas.
> 3. `docs/adr/` — 1–2 ADR yang terkait perubahan.
> 4. Ekor `CHANGELOG.md` bila perlu konteks rilis.

Aturan kerja:
- Ikuti DoD: `kode → test → formatter → entri CHANGELOG → update STATUS → ADR bila keputusan → npm run hub:report`.
- **Mulai fase**: tandai baris fase di `STATUS.md` menjadi 🔄 lalu jalankan `npm run hub:report` (status hub jadi *dikerjakan*). **Fase selesai**: tandai ✅ lalu `npm run hub:report` lagi (status hub jadi *selesai*). Tiap perubahan status memicu notifikasi (inbox + Web Push) di aplikasi portofolio.
- Conventional Commits; jangan commit rahasia (`.env` di-ignore).
- Setiap route/endpoint ber-mutasi harus punya test.
- Deploy ke EasyPanel dilakukan **programatik** (MCP/CLI), bukan klik UI manual — lihat `docs/runbooks/deploy.md`.
- Proyek ini **melapor ke hub portofolio**; lihat `scripts/hub-report.mjs`.
- Untuk proyek baru/scaffold, gunakan skill `aasaprojectkit`.

## Tujuan proyek
1. Belajar mengembangkan model AI secara bertahap (roadmap 12 minggu, lihat `ROADMAP.md`).
2. Membangun prototipe asisten BPS yang andal soal konsep/definisi statistik dan penyajian data resmi.
3. Hasil akhir didokumentasikan di GitHub sebagai portofolio.

## Pemilik proyek
- Bekerja di BPS, mengelola data. Pengguna Python yang terbiasa dengan data.
- Berkomunikasi dalam bahasa Indonesia santai. Jawab dan jelaskan dalam bahasa Indonesia; komentar kode dan nama variabel boleh bahasa Inggris.
- Tujuan jangka panjang: belajar sedalam mungkin, jadi **jelaskan kenapa**, bukan hanya apa. Setelah menulis kode non-trivial, jelaskan singkat cara kerjanya.
- Arah karier yang dipilih: **jalur lab model** (pengembangan, training, post-training, dan evaluasi model). Tekankan pemahaman konsep di balik training dan evals, bukan hanya penggunaan library.

## Lingkungan
- Laptop: ASUS VivoBook X409FJ, Windows 11, i7-8565U (4C/8T), RAM 8 GB, GPU MX230 2 GB.
- Belum ada PyTorch, CUDA, llama.cpp, atau Ollama (instal dari nol).
- Penyimpanan lega di drive E: dan F:. Simpan model, dataset, dan cache di E:/F:, bukan C:.
- **Jangan rencanakan training/fine-tuning di laptop ini.** Untuk GPU pakai Google Colab/Kaggle atau GPU sewaan. Di laptop: PyTorch versi CPU, model kecil, eksperimen ringan.
- RAM sangat terbatas: hindari proses yang memuat banyak data sekaligus; gunakan batch kecil, streaming, dan konteks maksimal 4096 untuk model lokal.

## Aturan kerja
- Kerjakan satu langkah kecil dalam satu waktu, jalankan, verifikasi, baru lanjut. Jangan membuat seluruh proyek sekaligus.
- Minta konfirmasi sebelum menginstal paket besar, mengunduh model berukuran GB, atau mengubah pengaturan sistem.
- Tulis dependensi di `requirements.txt` dan gunakan virtual environment (`.venv`).
- Setiap eksperimen disimpan di folder sendiri dengan catatan hasil singkat (`notes.md`): apa yang dicoba, hasil, apa yang dipelajari.
- Untuk setiap fitur RAG/asisten, tulis tes evaluasinya lebih dulu atau bersamaan, bukan belakangan.
- Kode harus sederhana dan terbaca; hindari abstraksi berlebihan. Ini proyek belajar.

## Aturan data (penting, instansi pemerintah)
- **Jangan mengirim data internal, tidak dipublikasikan, atau sensitif ke API eksternal.** Prototipe hanya memakai dokumen publik (publikasi, glosarium, metadata yang memang terbuka).
- Jangan menaruh kredensial, token, atau API key di dalam repo. Gunakan `.env` dan pastikan masuk `.gitignore`.
- Jika ragu apakah suatu data boleh dipakai, anggap tidak boleh dan tanyakan.

## Aturan untuk asisten BPS (prinsip desain)
- Konsep dan definisi: dijawab dari dokumen yang di-retrieve (RAG) dengan sitasi (judul, tahun, halaman/bagian).
- Angka dan data resmi: **hanya** dari output tool/API, tidak pernah dari ingatan model. Selalu sebutkan periode, wilayah, dan satuan.
- Jika sumber tidak ditemukan, jawab bahwa informasi tidak ditemukan. Jangan menebak.
- Fine-tuning hanya untuk perilaku (gaya bahasa, format, disiplin sitasi), bukan untuk menanamkan data.
- Evaluasi: test set 200–500 pertanyaan dengan jawaban acuan dari ahli substansi; ukur akurasi, ketepatan sitasi, tingkat halusinasi angka, dan penolakan yang benar.

## Struktur folder yang diinginkan
```
ai-project/
├── AGENTS.md
├── ROADMAP.md
├── requirements.txt
├── .env            # tidak di-commit
├── 01-pytorch-basics/
├── 02-mini-gpt/
├── 03-finetune-lora/
├── 04-rag-bps/
│   ├── data/       # dokumen publik saja
│   ├── src/
│   └── eval/
├── 05-serving/
└── 06-specialization/
```

## Gaya komunikasi
- Singkat dan konkret. Beri perintah terminal yang bisa langsung dijalankan (Windows PowerShell).
- Jika ada beberapa pilihan, rekomendasikan satu dan jelaskan alasannya dalam satu atau dua kalimat.
- Jujur soal keterbatasan: kalau sesuatu tidak realistis di laptop ini, katakan dan tawarkan alternatif.
