# Catatan — Fase 3: Fine-tuning LoRA (M5–6)

Format tiap entri: **apa yang dicoba · hasil · apa yang dipelajari**.

## Ringkasan
- [x] Pelajari konsep LoRA/QLoRA dan kenapa hemat memori
- [x] Jalankan di Google Colab (T4 GPU) — `finetune_openweight_colab.py` + `eval_format_colab.py`
- [x] Fine-tune model open-weight untuk satu tugas sempit (format jawaban ber-sitasi)
- [x] Bandingkan sebelum/sesudah dengan evaluasi sederhana (CPU & GPU)

## Cara menjalankan
```sh
# CPU (konsep):
.venv\Scripts\python 03-finetune-lora\01_lora_from_scratch.py
.venv\Scripts\python 03-finetune-lora\02_finetune_lora_minigpt.py

# GPU (Colab) — lihat COLAB.md:
# !python finetune_openweight_colab.py && python eval_format_colab.py

# Lokal (CPU) — adaptor dari Drive + base model di HF_HOME:
.venv\Scripts\python -c "import os; os.environ['HF_HOME']='E:/hf'"
.venv\Scripts\python 03-finetune-lora\03_ask_local.py "Apa definisi inflasi?"
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

- **Colab T4 · LoRA open-weight (2026-10-20)** — `finetune_openweight_colab.py` pada
  `Qwen/Qwen2.5-0.5B-Instruct`, Google Colab **Tesla T4** (15 GB). Hasil:
  - `trainable params: 2,162,688 / 496,195,456` = **0.4359%** (r=16 pada q/k/v/o_proj).
  - 10 langkah optimizer (4 contoh, batch 2, akumulasi 4, 10 epoch), **loss 3.913 → 2.705**.
  - Adaptor tersimpan (`adapter_model.safetensors`, 192 tensor); `lora_A` absmean ≈1.7e-2,
    `lora_B` absmean ≈4.1e-4 (nonzero tapi kecil — baru bergerak sedikit).
  - `eval_format_colab.py`: **skor format dasar 0/3 | LoRA 0/3**; jawaban dasar & LoRA
    **identik** (belum menyertakan sitasi, belum menolak "tidak ditemukan").
  Pelajaran: dengan hanya **4 contoh** dan **10 langkah**, LoRA belum mengubah perilaku
  terukur — `B` awal nol dan baru bergerak ~1e-4, jadi output ≈ model dasar. Untuk
  melatih perilaku (sitasi/abstain) perlu **lebih banyak & beragam contoh** dan **lebih
  banyak langkah**. Adaptor disalin ke Drive: `MyDrive/llm-bps/lora-bps-format`.

- **Temuan kompatibilitas Colab (penting)** — agar skrip jalan di Colab terbaru:
  1. `peft` terbaru menuntut `torchao>=0.16`, Colab punya 0.10 → error saat `get_peft_model`.
     Solusi: `!pip uninstall -y -q torchao` (torchao tak dipakai).
  2. TRL baru mengganti `SFTConfig(max_seq_length=...)` → `max_length`.
  3. T4 (compute capability 7.5) **tak mendukung bf16 matmul** → gunakan `torch.float16`.
     Awas: `torch.cuda.is_bf16_supported()` bisa **keliru True di T4**; cek
     `torch.cuda.get_device_capability()[0] >= 8` (Ampere+) — itulah yang dipakai `pick_dtype()`.
  Perbaikan **permanen** sudah masuk ke repo (`pick_dtype`, `max_length`, `processing_class`,
  AMP off di T4, + `!pip uninstall -y torchao` di docstring/`COLAB.md`), dan **terverifikasi
  di T4 tanpa patch in-memory**.

- **Lokal CPU (2026-10-20)** — adaptor dari Drive ditaruh di `./lora-bps-format`, base model
  diunduh ke `HF_HOME=E:\hf`, lalu `03_ask_local.py` memuat base + adaptor di CPU (fp32).
  Bukti adaptor benar-benar terpasang: `adaptor aktif: True`, `lora_B absmean(max) ≈ 5.9e-4`
  (cocok dengan Colab). Jawaban dasar vs LoRA **masih identik** — konsisten dengan adaptor yang
  belum terlatih cukup. Pelajaran: pipeline "latih di GPU → jalan di CPU lokal" **berfungsi**;
  jangan pakai `device_map="auto"` di CPU (membuat bobot adaptor nyangkut di `meta`).

## Latihan yang disarankan
1. `01`: naikkan `r` ke 4 lalu turunkan ke 1 — di titik mana kapasitas adaptasi jadi penghambat?
2. `02`: tambah pasangan `PAIRS`; apakah loss contoh tahanan makin rendah?
3. Colab: bandingkan `target_modules` attention-saja vs semua proyeksi; catat mutu vs parameter.

## Kaitan ke tujuan karier (jalur lab model)
LoRA/QLoRA adalah alat **post-training** harian di lab: menyesuaikan model dengan anggaran
GPU terbatas. Bagian yang paling "lab" bukan memakai `peft`, tetapi memahami pilihan
`r`, `alpha`, `target_modules`, dan kapan LoRA lebih tepat daripada full fine-tuning.
