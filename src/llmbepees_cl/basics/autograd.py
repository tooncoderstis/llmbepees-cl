"""Verifikasi autograd dengan turunan numerik (selisih tengah / central difference).

Ide: autograd menghitung gradien secara analitik lewat aturan rantai. Kita bisa
menguji hasilnya dengan aproksimasi numerik: menggeser tiap input sedikit
(±eps) dan melihat perubahan nilai fungsi.
"""

from __future__ import annotations

from collections.abc import Callable

import torch


def numerical_grad(
    fn: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    eps: float = 1e-3,
) -> torch.Tensor:
    """Gradien numerik `∇fn(x)` untuk `fn` yang mengembalikan skalar.

    Memakai selisih tengah `(f(x+eps) - f(x-eps)) / (2*eps)` per elemen.
    `x` dimodifikasi sementara lalu dikembalikan; kirim salinan bila perlu.
    """
    if fn(x).numel() != 1:
        raise ValueError("fn harus mengembalikan skalar (satu nilai)")

    grad = torch.zeros_like(x)
    flat_x = x.reshape(-1)
    flat_grad = grad.reshape(-1)
    for i in range(flat_x.numel()):
        original = flat_x[i].item()
        flat_x[i] = original + eps
        f_plus = fn(x)
        flat_x[i] = original - eps
        f_minus = fn(x)
        flat_x[i] = original  # pulihkan nilai asli
        flat_grad[i] = (f_plus - f_minus) / (2 * eps)
    return grad


__all__ = ["numerical_grad"]
