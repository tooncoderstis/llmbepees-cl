# Catatan — Fase 1: Dasar PyTorch (M1–2)

Format tiap entri: **apa yang dicoba · hasil · apa yang dipelajari**.

## Ringkasan
- [x] Tensor, autograd, `nn.Module`, training loop
- [x] MLP klasifikasi sederhana (data sintetis + MNIST subset)
- [x] Segarkan: aljabar linear, probabilitas, gradient descent

## Cara menjalankan skrip
```sh
.venv\Scripts\python 01-pytorch-basics\01_numpy_basics.py
.venv\Scripts\python 01-pytorch-basics\02_tensors.py
.venv\Scripts\python 01-pytorch-basics\03_autograd.py
.venv\Scripts\python 01-pytorch-basics\04_training_loop.py
.venv\Scripts\python 01-pytorch-basics\05_mlp_classifier.py
.venv\Scripts\python 01-pytorch-basics\06_mnist_optional.py   # butuh torchvision
```
Kode reusable ada di `src/llmbepees_cl/basics/` (diuji di `tests/`).

## Log
- **01 · NumPy warm-up** — matmul, softmax, regresi linear dengan gradient descent manual.
  Hasil: loss turun ke tingkat derau (≈0.0096) dan `w`/`b` belajar pulih (`w≈[1.497,-2.004,0.504]` vs `[1.5,-2,0.5]`).
  Pelajaran: gradient descent = ikuti arah turunan negatif; learning rate terlalu besar bisa divergen, terlalu kecil lambat.

- **02 · Tensor** — dtype (`float32` default), `reshape`, operasi per-elemen, `@` matmul, broadcasting `(3,1)+(4,)→(3,4)`.
  Pelajaran: broadcasting menyiarkan bentuk tanpa menyalin data; sumbu yang beda panjang tapi salah satunya 1 akan disiarkan.

- **03 · Autograd** — `requires_grad`, `.backward()`, `.grad`; dibandingkan turunan numerik (cocok: `[8.75, 12.0]`).
  Pelajaran: gradien **menumpuk** kalau tidak di-`zero_grad`; `torch.no_grad()` mematikan pelacakan saat evaluasi.

- **04 · Training loop** — urutan `zero_grad → forward → loss → backward → step`; akurasi uji 1.0 pada data sintetis.
  Pelajaran: batch kecil (32) cukup dan hemat memori; `nn.Sequential` adalah versi ringkas dari `nn.Module` buatan sendiri.

- **05 · Capstone (data sintetis)** — MLP 2→32→32→3, 60 epoch, akurasi uji 1.0; simpan/muat `state_dict` berhasil.
  Pelajaran: `state_dict` menyimpan **bobot**, bukan arsitektur — saat memuat, bangun ulang arsitektur yang sama.

- **06 · MNIST (opsional)** — 4000 contoh, MLP 784→64→10, 5 epoch → akurasi uji 0.889.
  Pelajaran: masalah nyata → akurasi tak sempurna; menambah epoch/lebar model menaikkan akurasi tapi mulai perlu arsitektur lebih baik (CNN).

## Latihan yang disarankan
1. Ubah `lr` di `01_numpy_basics.py` menjadi `0.5` dan `0.01` — amati pengaruh ke loss.
2. Di `05_mlp_classifier.py`, tambah `cluster_std` jadi `1.2` dan naikkan `n_centers` jadi `5` — apakah akurasi tetap 1.0?
3. Ganti `nn.ReLU()` jadi `nn.Tanh()` di `04_training_loop.py` — bandingkan kecepatan konvergensi.
