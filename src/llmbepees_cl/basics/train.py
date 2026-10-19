"""Loop training minimalis dan evaluasi — inti Fase 1.

Sengaja ditulis eksplisit (zero_grad -> forward -> loss -> backward -> step)
agar mekanisme training terlihat, bukan disembunyikan framework.
"""

from __future__ import annotations

import torch
from torch import nn

from llmbepees_cl.basics.models import MLP


def train_classifier(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    epochs: int = 50,
    lr: float = 0.1,
    batch_size: int = 32,
    seed: int | None = 0,
) -> list[float]:
    """Latih `model` klasifikasi dan kembalikan riwayat rata-rata loss per epoch.

    Batch kecil (`batch_size`) menjaga pemakaian memori tetap rendah.
    """
    if seed is not None:
        torch.manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    gen = torch.Generator().manual_seed(seed if seed is not None else 0)
    n = x.shape[0]
    history: list[float] = []

    for _ in range(epochs):
        perm = torch.randperm(n, generator=gen)
        running = 0.0
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            xb, yb = x[idx], y[idx]

            optimizer.zero_grad()  # buang gradien epoch sebelumnya
            logits = model(xb)  # forward pass
            loss = loss_fn(logits, yb)  # hitung error
            loss.backward()  # hitung gradien (autograd)
            optimizer.step()  # perbarui bobot

            running += loss.item() * xb.shape[0]
        history.append(running / n)

    return history


@torch.no_grad()
def evaluate_accuracy(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    """Akurasi klasifikasi dalam rentang 0..1."""
    predictions = model(x).argmax(dim=1)
    return (predictions == y).float().mean().item()


__all__ = ["MLP", "evaluate_accuracy", "train_classifier"]
