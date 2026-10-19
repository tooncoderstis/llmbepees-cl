# Panduan Belajar — Fase 1: Dasar PyTorch (M1–2)

Panduan ini mendampingi skrip di folder ini. Tujuannya bukan sekadar "kode jalan",
tapi memahami **kenapa** setiap langkah dilakukan. Kerjakan berurutan 01 → 06.

- **Prasyarat**: `.venv` terpasang dengan `torch` CPU.
  ```sh
  python -m venv .venv
  .venv\Scripts\python -m pip install -e ".[dev]"
  .venv\Scripts\python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
  ```
- **Kaitannya ke roadmap**: ini bagian "Minggu 1–2: Dasar PyTorch" di `ROADMAP.md`.
- **Kode reusable**: util di balik beberapa skrip ada di `src/llmbepees_cl/basics/` dan diuji di `tests/`.
- **Catatan hasil eksperimen**: lihat `notes.md` di folder ini.

Pola tiap bagian: **Apa yang dilakukan · Kenapa begitu · Jalankan sendiri · Latihan.**

---

## Skrip 01 — `01_numpy_basics.py` (pemanasan NumPy)

**Apa yang dilakukan.** Menunjukkan tiga hal mendasar: perkalian matriks (`@`),
fungsi softmax, dan regresi linear yang dilatih dengan **gradient descent manual**.

**Kenapa begitu.** Kita sengaja mulai dari NumPy (ADR-0001) supaya mekanisme yang
nanti "disembunyikan" PyTorch terlihat eksplisit. Gradient descent = mengikuti arah
turunan negatif dari loss: `w ← w − lr · ∂L/∂w`. Di sini gradien MSE dihitung dengan
tangan (`∂L/∂w = 2/n · Xᵀ·err`), bukan oleh autograd. Softmax mengubah skor mentah
(logits) jadi probabilitas yang jumlahnya 1; kita kurangi nilai maksimum dulu supaya
`exp` tidak meluap (stabil secara numerik).

**Jalankan sendiri.**
```sh
.venv\Scripts\python 01-pytorch-basics\01_numpy_basics.py
```
Perhatikan: loss turun lalu mendatar di sekitar tingkat derau (≈0.0096, karena data
memang diberi noise), dan `w`/`b` yang dipelajari mendekati nilai sebenarnya.

**Latihan.**
1. Ubah `lr = 0.1` menjadi `0.5` lalu `0.01` — amati loss (divergen? lambat?).
2. Tambah `noise scale` dari `0.1` ke `0.5` — sampai di mana loss mendatar?
3. Ganti loss MSE jadi MAE (`np.abs(err).mean()`) dan turunkan gradiennya sendiri.

---

## Skrip 02 — `02_tensors.py` (tipe & bentuk)

**Apa yang dilakukan.** Membuat tensor (skalar/vektor/matriks), melihat `dtype`,
`reshape`, operasi per-elemen, matmul `@`, dan **broadcasting**.

**Kenapa begitu.** Tensor ≈ array NumPy + dua tambahan: (1) bisa pindah device (GPU,
tapi kita CPU), dan (2) melacak gradien untuk autograd. Default `float32` (bukan
`float64`) karena lebih hemat memori dan cukup presisi untuk jaringan saraf.
Broadcasting menyiarkan bentuk yang berbeda **tanpa menyalin data**: `(3,1) + (4,)`
menjadi `(3,4)` karena setiap sumbu yang panjangnya 1 direntangkan.

**Jalankan sendiri.**
```sh
.venv\Scripts\python 01-pytorch-basics\02_tensors.py
```

**Latihan.**
1. Buat tensor `(2,1,3)` dan `(4,1)`; tebak hasil penjumlahannya, lalu cek.
2. `x = torch.arange(12)`, coba `.view(3,4)` vs `.reshape(3,4)` — bedanya apa?
3. Hitung `a @ b` (bukan `a @ b.T`) — mengapa bentuknya error?

---

## Skrip 03 — `03_autograd.py` (gradien otomatis)

**Apa yang dilakukan.** `requires_grad` + `.backward()` untuk mengisi `.grad`,
lalu membandingkannya dengan turunan numerik dari paket (`numerical_grad`).

**Kenapa begitu.** Autograd menelusuri graf operasi dan menerapkan aturan rantai
secara terprogram — hasilnya identik dengan turunan analitik. Gradien **menumpuk**
(tidak diganti) tiap `backward()`, sebab itu di training loop kita wajib
`optimizer.zero_grad()`. Saat evaluasi kita pakai `torch.no_grad()` untuk mematikan
pelacakan (hemat memori & lebih cepat). Turunan numerik (selisih tengah) dipakai
sebagai penguji independen: `(f(x+ε) − f(x−ε)) / (2ε)`.

**Jalankan sendiri.**
```sh
.venv\Scripts\python 01-pytorch-basics\03_autograd.py
```
Perhatikan gradien autograd dan numerik sama (`[8.75, 12.0]`) dan gradien menumpuk
jadi `8.0` setelah `backward()` dua kali.

**Latihan.**
1. Ubah `f` jadi `(t**2).sum()` — verifikasi `.grad` = `2t`.
2. Sisipkan `x.grad.zero_()` sebelum `backward()` kedua; apa yang berubah?
3. Perkecil `eps` jadi `1e-1` dan `1e-4` — kapan aproksimasi numerik memburuk?

---

## Skrip 04 — `04_training_loop.py` (anatomi training)

**Apa yang dilakukan.** Satu loop training utuh dengan `nn.Sequential`, mencetak
loss dan akurasi uji tiap 5 epoch.

**Kenapa begitu.** Inilah lima langkah yang ada di **setiap** training:
`zero_grad → forward → loss → backward → step`. `zero_grad` membuang gradien lama
(mengingat sifat menumpuk di Skrip 03), `backward` mengisi `.grad`, dan `step`
memperbarui bobot. Kita acak urutan data tiap epoch agar model tidak menghafal urutan;
batch kecil (32) menjaga memori tetap rendah.

**Jalankan sendiri.**
```sh
.venv\Scripts\python 01-pytorch-basics\04_training_loop.py
```

**Latihan.**
1. Ganti `nn.ReLU()` jadi `nn.Tanh()` — bandingkan kecepatan konvergensi.
2. Ubah `batch_size` jadi `1` dan `256` — apa efeknya ke loss/stabilitas?
3. Hapus `optimizer.zero_grad()` — apa yang terjadi? (prediksi dulu, lalu buktikan.)

---

## Skrip 05 — `05_mlp_classifier.py` (capstone, data sintetis)

**Apa yang dilakukan.** Pipeline lengkap: data → MLP 2→32→32→3 → latih → evaluasi →
**simpan/muat checkpoint** menggunakan util dari paket (bukan menulis loop ulang).

**Kenapa begitu.** Ini merangkum seluruh Fase 1. `state_dict` menyimpan **bobot**,
bukan arsitektur — jadi saat memuat, kamu harus membangun ulang arsitektur yang sama
lalu `load_state_dict`. Memisahkan bobot dan arsitektur itu sengaja: file jadi lebih
ringan dan eksplisit soal apa yang direkonstruksi.

**Jalankan sendiri.**
```sh
.venv\Scripts\python 01-pytorch-basics\05_mlp_classifier.py
```

**Latihan.**
1. Naikkan `cluster_std` ke `1.2` dan `n_centers` ke `5` — apakah akurasi tetap 1.0?
2. Ubah `hidden=(32, 32)` jadi `(8,)` — seberapa turun akurasinya?
3. Coba `torch.save(model, ...)` (seluruh objek) vs `state_dict` — apa bedanya?

---

## Skrip 06 — `06_mnist_optional.py` (MNIST, opsional)

**Apa yang dilakukan.** Melatih MLP pada data nyata: 4000 contoh MNIST (gambar 28×28
diratakan jadi vektor 784), MLP 784→64→10, 5 epoch.

**Kenapa begitu.** Data sintetis selalu "terlalu mudah". MNIST mengajarkan realita:
akurasi tidak sempurna, dan menambah epoch/lebar model cepat mentok — sinyal bahwa
arsitektur yang lebih tepat (mis. CNN yang memahami struktur gambar) diperlukan.
Dataset diunduh ke `01-pytorch-basics/data/mnist/` (di-ignore git) dengan subset kecil
supaya ramah CPU/RAM.

**Jalankan sendiri** (butuh ekstra `vision`).
```sh
.venv\Scripts\python 01-pytorch-basics\06_mnist_optional.py
```
Hasil acuan: akurasi uji ≈ 0.889 setelah 5 epoch.

**Latihan.**
1. Naikkan `subset` ke `10000` dan epoch ke `10` — seberapa akurasi naik?
2. Perlebar `hidden` jadi `(128, 128)` — bandingkan akurasi vs waktu.
3. Ganti normalisasi: bagi piksel dengan 255 sebelum training — pengaruhnya?

---

## Verifikasi

```sh
.venv\Scripts\ruff check .
.venv\Scripts\ruff format --check .
.venv\Scripts\pytest
```

Semua harus hijau. Setelah Fase 1, lanjut ke **Fase 2 — Transformer mini dari nol**
(`02-mini-gpt/`), tempat konsep matmul, softmax, dan training loop ini dipakai ulang
di dalam **self-attention**.
