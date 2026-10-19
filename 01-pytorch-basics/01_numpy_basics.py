"""Skrip 01 — Pemanasan NumPy: matmul, softmax, dan gradient descent manual.

Jalankan:
    .venv\\Scripts\\python 01-pytorch-basics\\01_numpy_basics.py

Kenapa mulai dari NumPy (ADR-0001)? Agar mekanisme yang nanti "disembunyikan"
PyTorch terlihat eksplisit: perkalian matriks, softmax, dan terutama **turunan
manual** dari loss. Di Skrip 03 kita akan minta autograd menghitung gradien ini
otomatis, lalu membandingkannya dengan hitungan tangan.
"""

from __future__ import annotations

import numpy as np

rng = np.random.default_rng(0)

# --- 1) Perkalian matriks: inti hampir semua layer neural network -----------
a = rng.normal(size=(2, 3))
b = rng.normal(size=(3, 4))
c = a @ b
print("matmul:", a.shape, "@", b.shape, "->", c.shape)


# --- 2) Softmax: ubah skor mentah (logits) jadi probabilitas ----------------
def softmax(z: np.ndarray) -> np.ndarray:
    shifted = z - z.max()  # kurangi nilai maksimum agar stabil secara numerik
    exp = np.exp(shifted)
    return exp / exp.sum()


logits = np.array([2.0, 1.0, 0.1])
probs = softmax(logits)
print("softmax:", np.round(probs, 3), "| jumlah =", round(float(probs.sum()), 3))


# --- 3) Regresi linear + gradient descent manual ----------------------------
# Kita cari w, b yang meminimalkan MSE antara (X @ w + b) dan y.
n, d = 200, 3
w_true = np.array([1.5, -2.0, 0.5])
b_true = 0.5

X = rng.normal(size=(n, d))
y = X @ w_true + b_true + rng.normal(scale=0.1, size=n)

w = np.zeros(d)
b = 0.0
lr = 0.1
for step in range(1, 201):
    pred = X @ w + b
    err = pred - y
    loss = np.mean(err**2)

    # Turunan MSE terhadap parameter (aturan rantai):
    #   dL/dw = 2/n * X^T err      dL/db = 2/n * sum(err)
    grad_w = 2.0 / n * X.T @ err
    grad_b = 2.0 / n * err.sum()

    w -= lr * grad_w  # langkah gradient descent
    b -= lr * grad_b

    if step % 50 == 0:
        print(f"step {step:3d}  loss {loss:.5f}")

print("\nnilai sebenarnya : w =", np.round(w_true, 3), "b =", b_true)
print("hasil belajar    : w =", np.round(w, 3), "b =", round(float(b), 3))
