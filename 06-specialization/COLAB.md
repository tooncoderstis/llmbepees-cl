# Menjalankan DPO open-weight di Google Colab

Preference tuning tidak dijalankan di laptop ini (lihat `AGENTS.md`).

## Langkah singkat
1. <https://colab.research.google.com> → **New notebook** → **Runtime → T4 GPU**.
2. Unggah `02_dpo_openweight_colab.py` (panel **Files**) atau mount Drive.
3. Pasang dependensi:
   ```bash
   !pip install -q "transformers>=4.45" "peft>=0.12" "trl>=0.11" "datasets>=2.20" "bitsandbytes>=0.43" accelerate
   !pip uninstall -y -q torchao  # Colab sering memuat torchao lama -> peft menolak; torchao tak dipakai
   ```
4. Jalankan:
   ```bash
   !python 02_dpo_openweight_colab.py
   ```
   `DPOTrainer` mencetak metrik (loss, reward margin, akurasi) per langkah.
5. Simpan adaptor ke Drive:
   ```python
   import shutil

   shutil.copytree("dpo-bps-behavior", "/content/drive/MyDrive/dpo-bps-behavior")
   ```

## Model & metode
- Default `Qwen/Qwen2.5-0.5B-Instruct` (kecil, multibahasa, Apache-2.0).
- **QLoRA 4-bit** (`bitsandbytes`) + LoRA (`peft`) → hanya sebagian kecil parameter dilatih,
  hemat VRAM. `β=0.1`.
- Reference implisit dari base model dengan adaptor LoRA dinonaktifkan (fitur PEFT di TRL).
- Dtype otomatis (`pick_dtype`): bf16 di GPU Ampere+, **fp16 di T4**. Di T4, AMP dimatikan
  (`bf16=False, fp16=False`) karena `GradScaler` gagal pada bf16.
- TRL terbaru butuh `processing_class=tokenizer` pada `DPOTrainer`.

## Aturan data
- Hanya teks publik; contoh di skrip ditulis sendiri. Jangan pakai data internal.
- DPO melatih **perilaku/preferensi** (sitasi, abstain), bukan menanamkan angka.

## Hasil
Catat di `notes.md`: model, `β`, langkah, metrik reward margin/akurasi, contoh jawaban
sebelum/sesudah.
