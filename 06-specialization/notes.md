# Catatan — Fase 6: Jalur lab model — post-training & evaluasi (M11–12)

Format tiap entri: **apa yang dicoba · hasil · apa yang dipelajari**.

## Ringkasan
- [x] Preference tuning: paham RLHF vs DPO; jalankan **DPO kecil dari nol** di CPU (MiniGPT)
- [x] Dasar RL untuk LLM: reward, policy gradient, PPO, verifiable reward (catatan `03_rl_concepts.md`)
- [x] Bangun benchmark kecil (200 pertanyaan) + analisis kegagalan (`eval/`)
- [x] Ringkasan paper (Transformer, Kaplan, Chinchilla, InstructGPT, DPO, Qwen2.5) di `papers/`
- [x] Jalankan DPO di Colab/GPU (`02_dpo_openweight_colab.py`, T4)
- [ ] Test set dengan jawaban acuan ahli substansi (sekarang masih draf otomatis)

## Cara menjalankan
```sh
.venv\Scripts\python 06-specialization\01_dpo_from_scratch.py
.venv\Scripts\python 06-specialization\eval\build_testset.py
.venv\Scripts\python 06-specialization\eval\run_eval.py
# GPU (Colab): lihat COLAB.md -> python 02_dpo_openweight_colab.py
```
Kode reusable: `src/llmbepees_cl/posttraining/` + `src/llmbepees_cl/evaluation/` (diuji di `tests/`).

## Log
- **DPO dari nol** — `loss = −log σ(β·[(logp_θ(y_w)−logp_θ(y_l)) − (logp_ref(y_w)−logp_ref(y_l))])`.
  Saat policy == reference, `loss = ln 2` (netral). Reference dibekukan sebagai jangkar.
  Pelajaran: DPO = preference tuning tanpa reward model & tanpa RL; satu loss seperti supervised.

- **01 · DPO pada MiniGPT** — pasangan preferensi (pilih memberi definisi atas menolak).
  Sebelum: loss 0.6931 (=ln 2), akurasi 0.0, reward (0,0). Sesudah 300 langkah: loss **0.0031**,
  akurasi **1.0**, reward dipilih/ditolak **(2.46, −3.43)**; parameter reference tak berubah.
  Pelajaran: DPO menggeser **preferensi** policy relatif ke reference — bukan menghafal jawaban baru.

- **Evaluasi Fase 6** — test set **200** pertanyaan dibangun deterministik dari korpus & tool
  (140 konsep × 5 variasi, 20 angka × 2 variasi, 40 abstain). Hasil: akurasi **100% (160/160)**,
  sitasi **100% (140/140)**, abstain **100% (40/40)**, angka-via-tool **100% (20/20)**, tanpa kegagalan.
  Pelajaran: harness bersama (`evaluation/`) dipakai ulang oleh Fase 4 & 6.

- **Colab T4 · DPO open-weight (2026-10-20)** — `02_dpo_openweight_colab.py` pada
  `Qwen/Qwen2.5-0.5B-Instruct`, QLoRA 4-bit (`bitsandbytes`) + LoRA `r=16`, `β=0.1`,
  Google Colab **Tesla T4**. Hasil: 5 langkah (4 pasang, batch 1, akumulasi 4, 5 epoch),
  **loss 0.6931 (=ln 2) → 0.3655** — turun dari titik netral seperti diharapkan; adaptor
  DPO tersimpan (`dpo-bps-behavior`). Disalin ke Drive: `MyDrive/llm-bps/dpo-bps-behavior`.
  Temuan kompatibilitas Colab: `peft`↔`torchao` (uninstall `torchao`), TRL `DPOTrainer`
  butuh `processing_class=tokenizer`, T4 tanpa bf16 → `torch.float16` dan **matikan AMP**
  (`bf16=False, fp16=False`) karena `GradScaler` gagal pada bf16 di T4. Awas:
  `torch.cuda.is_bf16_supported()` bisa keliru True di T4 → `pick_dtype()` memakai
  `torch.cuda.get_device_capability()[0] >= 8`. Perbaikan permanen terverifikasi di T4.
  Pelajaran: sama seperti LoRA, 4 pasangan × 5 langkah itu **demo mekanika**, belum
  mengubah perilaku terukur — perlu data preferensi jauh lebih banyak untuk efek nyata.

## Catatan jujur
- Hasil 100% itu **in-domain** (pertanyaan diturunkan dari korpus yang sama) — optimistis. Uji
  sesungguhnya butuh pertanyaan independen + jawaban acuan dari ahli substansi (lihat ADR-0004).
- DPO open-weight **sudah dijalankan di Colab T4** (QLoRA, loss ln2 → 0.37) — membuktikan
  alur GPU-nya jalan; tetapi jumlah pasangan contoh sangat kecil, jadi belum mengubah perilaku.

## Latihan yang disarankan
1. Ganti `chosen`/`rejected` DPO menjadi perilaku "sitasi wajib" — apakah reward margin naik?
2. Tambah kategori uji "angka tanpa periode" dan periksa abstain.
3. Jalankan `02_dpo_openweight_colab.py` di Colab; catat metrik & contoh jawaban sebelum/sesudah.
