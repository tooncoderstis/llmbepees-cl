# Menjalankan fine-tuning LoRA di Google Colab

Fine-tuning model open-weight **tidak** dijalankan di laptop ini (lihat `AGENTS.md`).
Skrip di folder ini disiapkan untuk Colab/Kaggle dengan GPU.

## Langkah singkat

1. Buka <https://colab.research.google.com> → **New notebook**.
2. **Runtime → Change runtime type → T4 GPU**.
3. Unggah folder ini (atau dua skrip `finetune_openweight_colab.py` dan
   `eval_format_colab.py`) lewat panel **Files**, atau mount Google Drive:
   ```python
   from google.colab import drive

   drive.mount("/content/drive")
   # lalu salin skrip dari Drive ke /content
   ```
4. Pasang dependensi (sekali per sesi):
   ```bash
   !pip install -q "transformers>=4.45" "peft>=0.12" "trl>=0.11" "datasets>=2.20" accelerate
   ```
5. Latih adaptor LoRA:
   ```bash
   !python finetune_openweight_colab.py
   ```
   `print_trainable_parameters()` akan menunjukkan hanya sebagian kecil parameter yang dilatih.
6. Evaluasi sebelum vs sesudah:
   ```bash
   !python eval_format_colab.py
   ```
7. Simpan adaptor ke Drive supaya tidak hilang saat sesi berakhir:
   ```python
   import shutil

   shutil.copytree("lora-bps-format", "/content/drive/MyDrive/lora-bps-format")
   ```

## Pilihan model

Default: **`Qwen/Qwen2.5-0.5B-Instruct`** — kecil (cukup untuk T4 gratis), multibahasa
(baik untuk bahasa Indonesia), lisensi Apache-2.0. Alternatif:
`meta-llama/Llama-3.2-1B-Instruct` (lisensi portal), `google/gemma-2-2b-it`,
atau `microsoft/Phi-3.5-mini-instruct`.

## Aturan data (penting)

- Hanya pakai dokumen/teks **publik**. `EXAMPLES` di skrip adalah contoh yang ditulis
  sendiri — ganti dengan contoh Anda, jangan menyalin data internal atau yang belum
  dipublikasikan.
- Fine-tuning di sini melatih **perilaku** (gaya, format, disiplin sitasi), bukan
  menanamkan data. Angka resmi tetap harus lewat tool (lihat prinsip di `AGENTS.md`).

## Hasil

Catat di `notes.md`: model, jumlah langkah, `trainable params %`, contoh jawaban
sebelum/sesudah, dan skor format. Adaptor (`*.safetensors`, dll.) di-`gitignore`.
