# Panduan Belajar — Fase 2: Transformer mini dari nol (M3–4)

Panduan ini mendampingi skrip di folder ini. Tujuannya bukan sekadar "kode jalan",
tapi memahami **kenapa** setiap langkah dilakukan. Kerjakan berurutan 01 → 05.

- **Prasyarat**: `.venv` terpasang dengan `torch` CPU (lihat `01-pytorch-basics/PANDUAN.md`).
- **Kaitannya ke roadmap**: bagian "Minggu 3–4: Transformer mini dari nol" di `ROADMAP.md`.
- **Kode reusable**: `src/llmbepees_cl/minigpt/` (diuji di `tests/test_minigpt_*.py`).
- **Catatan hasil eksperimen**: lihat `notes.md` di folder ini.
- Semua skrip hanya butuh CPU dan korpus kecil bawaan (`TINY_CORPUS`) — tanpa unduhan.

Pola tiap bagian: **Apa yang dilakukan · Kenapa begitu · Jalankan sendiri · Latihan.**

---

## Skrip 01 — `01_tokenizer.py` (teks → id)

**Apa yang dilakukan.** Membandingkan `CharTokenizer` (1 karakter = 1 token) dengan
`BPETokenizer` (menggabungkan pasangan yang sering muncul) pada satu kalimat.

**Kenapa begitu.** Model hanya melihat angka, jadi teks harus diubah jadi deret id.
Level karakter paling mudah dipahami dan kosakatanya kecil. BPE lebih realistis:
ia secara iteratif menggabungkan pasangan token paling sering, sehingga potongan
umum ("kan", "stat") menjadi satu token → deret lebih pendek dan model "melihat"
konteks lebih panjang untuk jumlah posisi yang sama. Karena itu tokenizer LLM asli
memakai BPE (atau variasinya). Keduanya wajib **roundtrip**: `decode(encode(x)) == x`.

**Jalankan sendiri.**
```sh
.venv\Scripts\python 02-mini-gpt\01_tokenizer.py
```
Perhatikan: 41 karakter menjadi 21 token BPE, dan decode tetap mengembalikan kalimat asli.

**Latihan.**
1. Naikkan `n_merges` ke 30 — seberapa pendek deretnya? Apakah decode masih utuh?
2. Encode teks baru berisi huruf yang tidak ada di kosakata latih — apa yang terjadi?
3. Kenapa `CharTokenizer` tidak punya masalah "unknown character" jika kosakata dibuat
   dari seluruh korpus?

---

## Skrip 02 — `02_attention.py` (siapa memperhatikan siapa)

**Apa yang dilakukan.** Menghitung matriks perhatian `CausalSelfAttention`, lalu uji:
mengganti token masa depan tidak mengubah output posisi sebelumnya.

**Kenapa begitu.** Attention menghitung skor query·key (kemiripan), menskalanya dengan
`1/sqrt(head_size)` agar softmax tidak jenuh, lalu memakai softmax itu sebagai bobot
rata-rata atas value. Untuk model bahasa, arah perhatian dibatasi **kausal**: posisi t
hanya boleh melihat 0..t — kalau tidak, model "mengintip jawaban" saat dilatih memprediksi
token berikutnya. Multi-head menjalankan beberapa attention paralel agar bisa menangkap
beberapa pola sekaligus.

**Jalankan sendiri.**
```sh
.venv\Scripts\python 02-mini-gpt\02_attention.py
```
Matriks perhatian berbentuk segitiga bawah; selisih output posisi 0–3 = 0.0.

**Latihan.**
1. Hapus baris `masked_fill` — apakah uji kausal gagal? Kenapa berbahaya saat training?
2. Ganti `n_head` dari 4 ke 1 dan 8 — model jadi lebih "sempit" atau "lebar"? Kenapa?
3. Hilangkan skala `1/sqrt(head_size)` — amati nilai matriks softmax (jenuh dekat 0/1?).

---

## Skrip 03 — `03_model.py` (merakit MiniGPT)

**Apa yang dilakukan.** Membangun `MiniGPT`, mencetak jumlah parameter dan arsitektur,
serta menelusuri bentuk tensor `idx [B,T] → logits [B,T,vocab]`.

**Kenapa begitu.** Blok transformer = attention (mencampur antar posisi) + feed-forward
(MLP per posisi), masing-masing dengan **residual** (`x + sublayer(ln(x))`) dan **pre-norm**
(LayerNorm sebelum sub-blok) yang membuat training lebih stabil. Embedding posisi ditambahkan
ke embedding token agar model tahu urutan (attention sendiri tidak punya urutan bawaan).

**Jalankan sendiri.**
```sh
.venv\Scripts\python 02-mini-gpt\03_model.py
```
Model kecil ini ≈ 0.11 juta parameter — beberapa orde lebih kecil dari LLM asli.

**Latihan.**
1. Ubah `n_embd` 64→128 dan `n_layer` 2→4; hitung ulang jumlah parameter (kira-kira 4x per dobel lebar?).
2. Naikkan `block_size` jadi 128 — apa pengaruhnya ke parameter embedding posisi?
3. Bila `seq_len` input melebihi `block_size`, error apa yang muncul? Kenapa dibatasi?

---

## Skrip 04 — `04_train.py` (melatih model bahasa)

**Apa yang dilakukan.** Melatih MiniGPT untuk memprediksi token berikutnya pada `TINY_CORPUS`,
mencetak loss latih & validasi, lalu menyimpan checkpoint (config + kosakata + bobot).

**Kenapa begitu.** Tugasnya self-supervised: target tiap posisi adalah token berikutnya (`y` = `x`
digeser satu), jadi tidak perlu label manusia. Loop-nya sama seperti Fase 1
(`zero_grad → forward → loss → backward → step`), hanya loss-nya cross-entropy antar token.
Kita pakai AdamW (standar untuk transformer). Checkpoint menyimpan `state_dict` **plus** konfigurasi
dan kosakata, supaya model bisa direkonstruksi persis saat generasi.

**Jalankan sendiri.**
```sh
.venv\Scripts\python 02-mini-gpt\04_train.py
```
Loss latih turun tajam (≈3.8 → ≈0.18). Loss validasi justru **naik** (≈4.4): model overfit
karena data sangat kecil.

**Latihan.**
1. Perbesar korpus (tambah paragraf di `TINY_CORPUS`) — apakah train dan validasi jadi lebih dekat?
2. Ubah `lr` ke `1e-2` dan `1e-3` — mana yang paling stabil?
3. Turunkan `n_layer` jadi 1 — apakah overfit berkurang?

---

## Skrip 05 — `05_generate.py` (menghasilkan teks)

**Apa yang dilakukan.** Memuat checkpoint, menyampel token secara autoregresif dengan
beberapa `temperature` dan `top_k`.

**Kenapa begitu.** Generasi = jalankan model, ambil distribusi token berikutnya, sampel satu
token, tempelkan ke konteks, ulangi. `temperature` membagi logits: <1 → distribusi lebih tajam
(aman/monoton), >1 → lebih rata (liar/kreatif). `top_k` memangkas pilihan ke k token teratas
sehingga token ekor yang aneh tidak terpilih. Konteks dibatasi `block_size` token terakhir.

**Jalankan sendiri** (setelah 04).
```sh
.venv\Scripts\python 02-mini-gpt\05_generate.py
```
`temperature=0.8` biasanya paling seimbang; `1.2` mulai mengulang frasa.

**Latihan.**
1. Ganti `prompt` dengan awal kalimat lain dari korpus — apakah lanjutannya masuk akal?
2. Naikkan `n_new` ke 300 — kapan teks mulai melenceng dari topik?
3. Bandingkan `top_k=1` (selalu token teratas) vs `top_k=None` — mana yang lebih berulang?

---

## Verifikasi

```sh
.venv\Scripts\ruff check .
.venv\Scripts\ruff format --check .
.venv\Scripts\pytest
```

Semua harus hijau. Setelah Fase 2, lanjut ke **Fase 3 — Fine-tuning LoRA** (`03-finetune-lora/`),
yang dijalankan di Colab/GPU (bukan laptop ini), memanfaatkan konsep transformer di atas untuk
menyesuaikan model open-weight pada satu tugas sempit.
