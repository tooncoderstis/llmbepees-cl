"""Skrip 02 — Tensor PyTorch: dtype, bentuk, broadcasting, dan matmul.

Tensor = array multi-dimensi seperti NumPy, tapi menambah dua hal penting:
(1) bisa berjalan di GPU (di sini CPU), dan (2) melacak gradien untuk autograd
(dibahas di Skrip 03).

Jalankan:
    .venv\\Scripts\\python 01-pytorch-basics\\02_tensors.py
"""

from __future__ import annotations

import torch

# --- 1) Membuat tensor: skalar, vektor, matriks -----------------------------
scalar = torch.tensor(3.0)
vector = torch.tensor([1.0, 2.0, 3.0])
matrix = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
print("scalar", scalar.shape, "| vector", vector.shape, "| matrix", matrix.shape)

# --- 2) dtype default float32 & konversi ------------------------------------
print("default dtype:", matrix.dtype)
print("pakai float64:", matrix.to(torch.float64).dtype)

# --- 3) reshape: mengubah bentuk tanpa mengubah data ------------------------
x = torch.arange(12)
print("arange(12):", x.shape, "-> reshape(3,4):", x.reshape(3, 4).shape)

# --- 4) Operasi elemen & perkalian matriks ----------------------------------
a = torch.ones(2, 3)
b = torch.full((2, 3), 2.0)
print("a + b (per elemen):", (a + b)[0].tolist())
print("a @ b.T :", (a @ b.T).shape)  # (2,3) @ (3,2) -> (2,2)

# --- 5) Broadcasting: bentuk berbeda otomatis disiarkan ---------------------
col = torch.tensor([[1.0], [2.0], [3.0]])  # (3,1)
row = torch.tensor([10.0, 20.0, 30.0, 40.0])  # (4,)
print("(3,1) + (4,):", (col + row).shape)  # -> (3,4)
print(col + row)

# --- 6) Device: di sini selalu cpu ------------------------------------------
print("device:", matrix.device)
