# Ringkasan paper (M11–12)

Catatan singkat tiap paper **dengan kata sendiri**: ide inti + kenapa penting untuk
proyek ini. Baca satu per minggu, ringkas ulang dengan bahasamu.

## 1. Attention Is All You Need (Vaswani dkk., 2017)
- **Ide inti**: arsitektur **Transformer** — sepenuhnya berbasis *self-attention* (tanpa RNN/konvolusi), memakai multi-head attention, positional encoding, residu + LayerNorm.
- **Kenapa penting**: fondasi semua LLM modern; kita mengimplementasikannya dari nol di Fase 2 (`minigpt/attention.py`, `model.py`).
- Tautan: <https://arxiv.org/abs/1706.03762>

## 2. Scaling Laws for Neural Language Models (Kaplan dkk., 2020)
- **Ide inti**: loss menurun sebagai **hukum pangkat** yang mulus terhadap ukuran model (N), data (D), dan komputasi (C). Kapasitas model lebih menentukan di awal, data kemudian.
- **Kenapa penting**: alasan "kenapa besar bisa lebih baik"; kita mengulang versi kecilnya di `05-serving/02_scaling.py`.
- Tautan: <https://arxiv.org/abs/2001.08361>

## 3. Training Compute-Optimal LLMs / "Chinchilla" (Hoffmann dkk., 2022)
- **Ide inti**: untuk anggaran komputasi tetap, **parameter dan data harus tumbuh seimbang** — model-model awal (termasuk perbandingan di paper Kaplan) **kekurangan data**. Aturan praktis: ~20 token per parameter.
- **Kenapa penting**: mengoreksi "cukup besarkan model"; menjelaskan kenapa kita fokus data & token, bukan hanya parameter.
- Tautan: <https://arxiv.org/abs/2203.15556>

## 4. InstructGPT / Training language models to follow instructions (Ouyang dkk., 2022)
- **Ide inti**: resep **RLHF** — SFT → reward model dari preferensi → PPO dengan penalti KL. Model jadi lebih patuh dan lebih disukai manusia.
- **Kenapa penting**: dasar "post-training"; memperkenalkan gagasan menyelaraskan perilaku (bukan menambah data ke bobot).
- Tautan: <https://arxiv.org/abs/2203.02155>

## 5. Direct Preference Optimization — DPO (Rafailov dkk., 2023)
- **Ide inti**: reward model optimal punya bentuk tertutup, jadi RLHF dapat **disederhanakan** menjadi loss seperti klasifikasi preferensi langsung pada policy (reference dibekukan, `β` mengatur deviasi).
- **Kenapa penting**: preference tuning stabil & murah; kita implementasikan dari nol di `posttraining/dpo.py`.
- Tautan: <https://arxiv.org/abs/2305.18290>

## 6. Qwen2.5 Technical Report (Qwen Team, 2024)
- **Ide inti**: laporan model open-weight kecil–besar (termasuk 0.5B–7B), data & tahap training (pretraining → SFT → preference), evaluasi luas, lisensi permisif.
- **Kenapa penting**: contoh nyata "cara lab melaporkan model"; basis untuk fine-tuning/DPO di skrip Colab Fase 3 & 6.
- Tautan: <https://arxiv.org/abs/2412.15115>

## Kebiasaan
- Satu paper per minggu; ringkas: **masalah · ide · hasil · batasan · kaitan ke proyek**.
- Jangan hafal angka — pahami **mekanisme** dan **kapan** metodenya dipakai.
