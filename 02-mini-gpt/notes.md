# Catatan — Fase 2: Transformer mini dari nol (M3–4)

Format tiap entri: **apa yang dicoba · hasil · apa yang dipelajari**.

## Ringkasan
- [x] Tokenisasi karakter, lalu BPE sederhana
- [x] Self-attention, blok transformer, GPT mini
- [x] Latih pada teks kecil (CPU cukup, beberapa juta parameter)

## Cara menjalankan skrip
```sh
.venv\Scripts\python 02-mini-gpt\01_tokenizer.py
.venv\Scripts\python 02-mini-gpt\02_attention.py
.venv\Scripts\python 02-mini-gpt\03_model.py
.venv\Scripts\python 02-mini-gpt\04_train.py
.venv\Scripts\python 02-mini-gpt\05_generate.py   # butuh checkpoint dari 04
```
Kode reusable ada di `src/llmbepees_cl/minigpt/` (diuji di `tests/test_minigpt_*.py`).

## Log
- **01 · Tokenisasi** — `CharTokenizer` (1 karakter = 1 token) vs `BPETokenizer` pada kalimat
  "statistik menyajikan data statistik resmi": 41 token (char) → 21 token (BPE, 10 merge).
  Pelajaran: BPE memperpendek deret dengan menggabungkan pasangan yang sering ("stat", "istik"),
  dan kedua tokenizer tetap bisa *roundtrip* (`decode(encode(x)) == x`).

- **02 · Self-attention kausal** — matriks perhatian berbentuk **segitiga bawah** (posisi t
  hanya melihat 0..t); setelah token masa depan (posisi 4–5) diubah, output posisi 0–3 identik
  (selisih 0.0) sedangkan 4–5 berubah (0.23).
  Pelajaran: masker kausal + skala `1/sqrt(head_size)` adalah yang membuat attention aman untuk
  prediksi token berikutnya dan stabil secara numerik.

- **03 · MiniGPT** — model `n_embd=64, n_head=4, n_layer=2, block_size=64`, kosakata 42 → **109.184
  parameter**. Bentuk tensor: `idx [B,T]` → `logits [B,T,vocab]`; `loss=None` tanpa target.
  Pelajaran: seluruh model cuma rangkaian `Linear` + embedding + LayerNorm — tidak ada keajaiban,
  hanya komposisi matmul/softmax yang diulang.

- **04 · Training** — 800 langkah, batch 16, `block_size=64`, AdamW `lr=3e-3` pada `TINY_CORPUS`
  (~2.8k token): loss latih **3.77 → 0.18**; loss validasi **4.45** (acuan acak `ln(42)≈3.74`).
  Pelajaran: model kecil pun **overfit** data sekecil ini (train ≪ validasi). Menambah data/embargo
  atau memperkecil model adalah arah perbaikan; ini cikal bakal topik *regularization* di Fase 5.

- **05 · Generasi** — memuat checkpoint, menyampel dengan temperatur 0.5/0.8/1.2 dan `top_k=5`.
  Hasil (contoh, prompt "Data statistik adalah "):
  - temp 0.5: teks pendek, agak monoton ("Data statistik adalah ilmu tenyatakan bagian ...")
  - temp 0.8: paling masuk akal ("... adalah ilmu tenyatakan ukuran yang dipakai pada setiap angka data.")
  - temp 1.2: liar, mulai mengulang ("yang yang menyatakan ...")
  Pelajaran: **temperature** menukar keragaman vs koherensi; `top_k` memangkas ekor distribusi.
  Karena model overfit, output cenderung menempel frasa korpus — musuh utama kualitas bahasa.

## Latihan yang disarankan
1. Perbesar `TINY_CORPUS` (mis. tambah paragraf) lalu ulang 04 — apakah selisih train/validasi menyusut?
2. Ubah `n_layer` dari 2 menjadi 1 dan 4 di `03_model.py` — lihat jumlah parameter & waktu training.
3. Di `05_generate.py`, naikkan `top_k` (mis. 20) dan bandingkan dengan tanpa `top_k`.
4. Ganti `CharTokenizer` jadi `BPETokenizer` di 04/05 — apakah loss per token berubah? Kenapa?
