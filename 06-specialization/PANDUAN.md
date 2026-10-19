# Panduan Belajar — Fase 6: Jalur lab model — post-training & evaluasi (M11–12)

Fase penutup. Tujuannya menyentuh **jalur lab model** dengan cara yang jujur: memahami
**preference tuning (DPO)** dan **RL untuk LLM**, lalu membangun **benchmark evaluasi sendiri**
dan menganalisis kegagalan. Kaitannya ke `ROADMAP.md`: "Minggu 11–12".

- **Kode reusable**: `src/llmbepees_cl/posttraining/` (DPO) dan `src/llmbepees_cl/evaluation/` (harness).
- **Skrip**: `01_dpo_from_scratch.py` (CPU), `02_dpo_openweight_colab.py` (GPU), `03_rl_concepts.md` (teori), `eval/` (benchmark).
- **Ringkasan paper**: `papers/README.md`.
- Prinsip: DPO melatih **perilaku/preferensi**, bukan menanamkan angka; angka tetap hanya dari tool.

Pola tiap bagian: **Apa yang dilakukan · Kenapa begitu · Jalankan sendiri · Latihan.**

---

## Konsep: dari RLHF ke DPO (`03_rl_concepts.md`)

- **RLHF**: SFT → reward model → PPO dengan penalti KL. Kuat tapi rumit & tidak stabil.
- **DPO**: hilangkan reward model + RL; optimalkan policy langsung pada data preferensi
  dengan membandingkan rasio log-prob policy vs reference (`loss = −log σ(β·Δ)`).
- **Verifiable reward**: untuk tugas dengan cek objektif (format, sitasi, angka dari tool),
  reward bisa dihitung programatik → dasar RL modern (GRPO) yang relevan untuk asisten BPS.

---

## Skrip 01 — `01_dpo_from_scratch.py` (CPU)

**Apa yang dilakukan.** Melatih MiniGPT dengan DPO pada pasangan preferensi (lebih memilih
memberi definisi daripada menolak). Menampilkan loss & reward sebelum/sesudah.

**Kenapa begitu.** Membuktikan mekanika: di awal policy == reference → `loss = ln 2 ≈ 0.693`;
setelah training, reward jawaban dipilih naik relatif yang ditolak, sedangkan reference
(tersalin beku) tak berubah.

**Jalankan sendiri.**
```sh
.venv\Scripts\python 06-specialization\01_dpo_from_scratch.py
```
Contoh: loss `0.693 → ~0.003`, akurasi preferensi `0 → 1.0`, reward (dipilih, ditolak) `(0,0) → (2.5, -3.4)`.

**Latihan.** 1) Ubah `beta` (0.01 / 0.5) — apa efeknya ke reward? 2) Tukar `chosen`/`rejected` — apa yang terjadi? 3) Tambah pasangan preferensi.

---

## Skrip 02 — `02_dpo_openweight_colab.py` + `COLAB.md` (GPU)

**Apa yang dilakukan.** DPO/QLoRA 4-bit pada `Qwen2.5-0.5B-Instruct` via TRL `DPOTrainer`,
melatih perilaku ber-sitasi & abstain dari contoh preferensi publik.

**Kenapa begitu.** Ini praktik sesungguhnya (model nyata + library nyata). Reference implisit
dari base model (adaptor LoRA dinonaktifkan).

**Jalankan sendiri.** Ikuti `COLAB.md` (Runtime → T4 GPU). Tidak dijalankan di laptop.

**Latihan.** 1) Naikkan `beta`/langkah — apakah reward margin membaik? 2) Tambah pasangan yang menekankan "jangan mengarang angka". 3) Bandingkan DPO vs SFT pada tugas yang sama.

---

## Benchmark — `eval/build_testset.py` + `eval/run_eval.py`

**Apa yang dilakukan.** Membuat test set ~200 pertanyaan (konsep/angka/abstain) secara
deterministik dari korpus & tool, menjalankan asisten, menghitung metrik, dan mengelompokkan
kegagalan per jenis.

**Kenapa begitu.** PRD menuntut metrik eksplisit. Harness dipakai bersama (DRY) oleh Fase 4 & 6.
`must_contain` diambil otomatis dari teks korpus → **draf**; jawaban acuan final perlu validasi
ahli substansi (lihat ADR-0004).

**Jalankan sendiri.**
```sh
.venv\Scripts\python 06-specialization\eval\build_testset.py
.venv\Scripts\python 06-specialization\eval\run_eval.py
```
Acuan: 200 pertanyaan; akurasi 100% (160/160), sitasi 100% (140/140), abstain 100% (40/40), angka-via-tool 100% (20/20). (In-domain → optimistis.)

**Latihan.** 1) Tambah pertanyaan "jebakan" (definisi mirip: populasi vs sampel). 2) Tambah kategori baru (mis. "fakta angka tanpa periode") dan jalankan analisis kegagalan. 3) Ganti mesin jawab dengan LLM lalu bandingkan.

---

## Ringkasan paper — `papers/README.md`
Transformer · Kaplan · Chinchilla · InstructGPT · DPO · Qwen2.5. Baca satu per minggu.

---

## Verifikasi
```sh
.venv\Scripts\ruff check .
.venv\Scripts\ruff format --check .
.venv\Scripts\pytest
```

Setelah Fase 6: proyek 12 minggu tuntas. Langkah lanjut ada di `ROADMAP.md` bagian
"Setelah 12 minggu" (reproduksi paper, kontribusi open-source, tulisan teknis, pilih bidang).
