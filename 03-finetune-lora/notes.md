# Catatan — Fase 3: Fine-tuning LoRA (M5–6)

Format tiap entri: **apa yang dicoba · hasil · apa yang dipelajari**.

## Ringkasan
- [x] Pelajari konsep LoRA/QLoRA dan kenapa hemat memori
- [ ] Jalankan di Google Colab/Kaggle (menunggu sesi GPU) — skrip & panduan siap
- [ ] Fine-tune model open-weight untuk satu tugas sempit (format jawaban ber-sitasi)
- [x] Bandingkan sebelum/sesudah dengan evaluasi sederhana (versi CPU; versi GPU menunggu)

## Cara menjalankan
```sh
# CPU (konsep):
.venv\Scripts\python 03-finetune-lora\01_lora_from_scratch.py
.venv\Scripts\python 03-finetune-lora\02_finetune_lora_minigpt.py

# GPU (Colab) — lihat COLAB.md:
# !python finetune_openweight_colab.py && python eval_format_colab.py
```
Kode reusable LoRA ada di `src/llmbepees_cl/lora/` (diuji di `tests/test_lora.py`).

## Log
- **Konsep** — `ΔW = (α/r)·B·A`, `W` beku, `B` mulai nol → output awal = model dasar.
  Pelajaran: parameter latih turun dari `in×out` ke `r×(in+out)`; tanpa gradien/optimizer
  state untuk `W`, memori training drop drastis. QLoRA menambah kuantisasi 4-bit pada `W`.

- **01 · LoRA dari nol** — jaringan dasar dilatih tugas A (loss 0.012), lalu adaptasi ke tugas B
  dengan LoRA `r=2`: **2 lapisan diganti**, dilatih **88 / 300** parameter (**29.3%**),
  loss B **21.88 → 0.48**, `bobot dasar tak berubah: True`.
  Pelajaran: matriks berpangkat rendah cukup untuk berpindah tugas; merge `W+BA` = bobot efektif.

- **02 · LoRA pada MiniGPT** — model dasar pada `TINY_CORPUS`, tugas sempit format
  "T: <konsep>\\nJ: <definisi>". Dilatih **4.096 / 113.408** parameter (**3.6%**, 8 lapisan attention).
  Loss format (latih) **2.54 → 0.76**; loss **contoh tahanan (tak dilatih) 2.26 → 1.98**.
  Pelajaran: LoRA menggeneralisasi ke contoh baru (loss turun), membuktikan adaptasi bukan sekadar hafalan.
  Catatan jujur: teks hasil generasi masih kacau karena model karakter ini sangat kecil — angka loss
  adalah bukti utamanya; kualitas bahasa sebenarnya diuji di skrip Colab.

- **Colab (belum dijalankan)** — `finetune_openweight_colab.py` (Qwen2.5-0.5B-Instruct, `peft`+`trl`)
  dan `eval_format_colab.py` (bandingkan sitasi & perilaku "tidak ditemukan" sebelum/sesudah).
  Dijalankan pengguna di GPU; hasil akan dicatat di sini.

## Latihan yang disarankan
1. `01`: naikkan `r` ke 4 lalu turunkan ke 1 — di titik mana kapasitas adaptasi jadi penghambat?
2. `02`: tambah pasangan `PAIRS`; apakah loss contoh tahanan makin rendah?
3. Colab: bandingkan `target_modules` attention-saja vs semua proyeksi; catat mutu vs parameter.

## Kaitan ke tujuan karier (jalur lab model)
LoRA/QLoRA adalah alat **post-training** harian di lab: menyesuaikan model dengan anggaran
GPU terbatas. Bagian yang paling "lab" bukan memakai `peft`, tetapi memahami pilihan
`r`, `alpha`, `target_modules`, dan kapan LoRA lebih tepat daripada full fine-tuning.
