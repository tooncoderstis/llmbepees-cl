# Panduan Belajar — Fase 3: Fine-tuning LoRA (M5–6)

Tujuannya memahami **kenapa** LoRA/QLoRA hemat memori dan **bagaimana** ia bekerja,
lalu mempraktikkannya: sedikit di laptop (CPU, model mini dari Fase 2) dan yang
sebenarnya di **Colab/GPU** (model open-weight). Kaitannya ke `ROADMAP.md`:
"Minggu 5–6: Fine-tuning LoRA".

> Aturan laptop: jangan latih/fine-tune model besar di sini. CPU hanya untuk konsep.

Pola tiap bagian: **Apa yang dilakukan · Kenapa begitu · Jalankan sendiri · Latihan.**

---

## Konsep: kenapa LoRA?

Saat full fine-tuning, kita memperbarui **semua** bobot `W` (tiap matriks `[in × out]`).
Itu butuh memori untuk gradien + optimizer state (AdamW menyimpan 2 nilai per parameter!)
sehingga bisa 3–4× ukuran model. LoRA membekukan `W` dan hanya melatih perubahan
berpangkat rendah:

```
W_efektif = W + (α/r) · B · A        A: [r, in],  B: [out, r],  r ≪ min(in, out)
```

Jumlah parameter latih turun dari `in×out` menjadi `r×(in+out)`. `B` dimulai dari nol
sehingga `ΔW = 0` di awal — model mulai persis dari perilaku pra-latih. Karena `W`
beku, gradien tidak perlu dihitung/disimpan untuknya → memori optimizer jauh lebih kecil.
**QLoRA** menambahkan kuantisasi 4-bit pada `W` (hemat makin banyak, sedikit menurunkan
presisi). Setelah selesai, `BA` bisa "dilipat" ke `W` (merge) sehingga inference tanpa
overhead. `r` mengatur kapasitas adaptasi; `α` mengatur skala kontribusi LoRA.

---

## Skrip 01 — `01_lora_from_scratch.py` (mekanika di CPU)

**Apa yang dilakukan.** Melatih jaringan dasar untuk tugas A, lalu memakai LoRA untuk
berpindah ke tugas B; mencetak jumlah parameter yang dilatih, loss sebelum/sesudah, dan
membuktikan bobot dasar tidak berubah.

**Kenapa begitu.** Ini versi paling telanjang dari LoRA: kita lihat sendiri bahwa output
awal LoRA identik dengan model dasar (`B=0`), hanya `A`/`B` yang punya gradien, dan model
bisa berpindah tugas walau parameter yang dilatih sedikit.

**Jalankan sendiri.**
```sh
.venv\Scripts\python 03-finetune-lora\01_lora_from_scratch.py
```
Loss tugas B turun (≈21.9 → ≈0.5) dengan ~29% parameter, dan `bobot dasar tak berubah: True`.

**Latihan.**
1. Ubah `r` dari 2 ke 1 dan 4 — kapan loss tidak bisa turun lagi? Kenapa?
2. Naikkan `alpha` — apa efeknya ke kecepatan belajar adaptor?
3. Coba `target_modules=("0",)` saja (hanya lapisan pertama) — cukupkah?

---

## Skrip 02 — `02_finetune_lora_minigpt.py` (LoRA pada MiniGPT)

**Apa yang dilakukan.** Melatih MiniGPT (Fase 2) pada korpus statistik, mengukur loss pada
contoh format "T: … \\nJ: …" yang **ditahan**, menerapkan LoRA, fine-tune hanya adaptor, lalu
mengukur ulang loss contoh tahanan + melihat contoh jawaban.

**Kenapa begitu.** Ini menyambung konsep ke model bahasa: kita membekukan seluruh model dan
melatih matriks berpangkat rendah di attention. Perhatikan tiga hal: (1) hanya ~3.6% parameter
dilatih, (2) loss pada contoh yang **tidak** ikut dilatih ikut turun, (3) bobot dasar utuh.
Model ini sengaja sangat kecil — tujuannya mekanika, bukan kualitas bahasa (untuk itu lihat skrip Colab).

**Jalankan sendiri.**
```sh
.venv\Scripts\python 03-finetune-lora\02_finetune_lora_minigpt.py
```
Loss contoh tahanan turun (≈2.26 → ≈1.98) dengan ≈4k dari ≈113k parameter dilatih.

**Latihan.**
1. Turunkan `r` ke 1 — masihkah loss contoh tahanan turun?
2. Tambah pasangan di `PAIRS` lalu ulang — apakah loss tahanan makin rendah?
3. Latih model dasar lebih lama (`steps`) — apakah LoRA jadi lebih sulit mengubah perilaku?

---

## Skrip Colab — `finetune_openweight_colab.py` + `eval_format_colab.py`

**Apa yang dilakukan.** Fine-tune `Qwen2.5-0.5B-Instruct` dengan library `peft`/`trl` untuk
tugas berformat sitasi gaya BPS, lalu membandingkan jawaban model dasar vs model + adaptor.

**Kenapa begitu.** Ini praktik sesungguhnya: model nyata, library nyata, GPU nyata. Perhatikan
`print_trainable_parameters()` (biasanya <1% parameter). Evaluasi sederhana mengecek apakah
jawaban menyertakan sitasi dan menolak saat sumber tidak ada.

**Jalankan sendiri.** Ikuti `COLAB.md` (Runtime → T4 GPU). Tidak dijalankan di laptop.

**Latihan.**
1. Ganti `target_modules` ke semua proyeksi termasuk `gate_proj`/`up_proj`/`down_proj` — bandingkan hasil.
2. Naikkan `r` dari 16 ke 64 — apakah kualitas naik? Berapa biaya parameter-nya?
3. Bandingkan LoRA (`peft`) vs full fine-tuning pada model yang sama — memori vs mutu.

---

## Verifikasi

```sh
.venv\Scripts\ruff check .
.venv\Scripts\ruff format --check .
.venv\Scripts\pytest
```

Setelah Fase 3, lanjut ke **Fase 4 — RAG + tool calling** (`04-rag-bps/`): prototipe asisten
BPS yang mengambil jawaban dari dokumen publik ber-sitasi dan hanya menyajikan angka lewat tool.
