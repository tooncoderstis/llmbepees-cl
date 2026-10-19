"""Skrip 03 — Autograd: gradien otomatis, dibandingkan dengan hitungan tangan.

Di Skrip 01 kita menurunkan gradien MSE secara manual. Autograd melakukan hal
yang sama secara otomatis dengan menelusuri graf operasi ("aturan rantai"
terprogram). Di sini kita buktikan hasilnya cocok.

Jalankan:
    .venv\\Scripts\\python 01-pytorch-basics\\03_autograd.py
"""

from __future__ import annotations

import torch

from llmbepees_cl.basics import numerical_grad

# --- 1) Fungsi skalar sederhana f(t) = sum(t^3) + 2*t0 ----------------------
x = torch.tensor([1.5, -2.0], requires_grad=True)


def f(t: torch.Tensor) -> torch.Tensor:
    return (t**3).sum() + 2 * t[0]


y = f(x)
y.backward()  # isi x.grad
print("f(x) =", round(y.detach().item(), 3))
print("gradien autograd:", [round(v, 3) for v in x.grad.tolist()])

# --- 2) Bandingkan dengan turunan numerik dari paket ------------------------
numeric = numerical_grad(f, x.detach().clone(), eps=1e-2)
print("gradien numerik :", [round(v, 3) for v in numeric.tolist()])

# --- 3) Akumulasi gradien: gradien MENUMPUK bila tidak di-zero --------------
print("\n-- akumulasi gradien --")
x2 = torch.tensor([2.0], requires_grad=True)
(x2**2).sum().backward()
print("setelah backward ke-1:", x2.grad.item())
(x2**2).sum().backward()
print("setelah backward ke-2:", x2.grad.item(), "(menumpuk, bukan diganti)")

# --- 4) no_grad: matikan pelacakan saat evaluasi (hemat memori) -------------
with torch.no_grad():
    z = x2 * 3
print("\nno_grad -> requires_grad:", z.requires_grad)
