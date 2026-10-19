"""Loop training model bahasa: satu langkah = prediksi token berikutnya.

Sama seperti Fase 1 (`zero_grad -> forward -> loss -> backward -> step`), tetapi
loss-nya cross-entropy antar token: model belajar menebak token selanjutnya pada
setiap posisi sekaligus. Dipakai AdamW, optimizer standar untuk transformer.
"""

from __future__ import annotations

import torch
from torch import nn

from llmbepees_cl.minigpt.data import get_batch


def train_language_model(
    model: nn.Module,
    train_data: torch.Tensor,
    *,
    steps: int = 500,
    batch_size: int = 16,
    block_size: int = 64,
    lr: float = 3e-3,
    seed: int | None = 0,
    log_every: int = 0,
) -> list[float]:
    """Latih `model` pada `train_data` (tensor id 1D) selama `steps` iterasi.

    Mengembalikan riwayat loss per langkah (list sepanjang `steps`). Batch kecil
    menjaga pemakaian memori rendah di laptop.
    """
    if seed is not None:
        torch.manual_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(seed if seed is not None else 0)

    model.train()
    history: list[float] = []
    for step in range(1, steps + 1):
        xb, yb = get_batch(train_data, block_size, batch_size, gen)

        optimizer.zero_grad(set_to_none=True)
        _, loss = model(xb, yb)
        loss.backward()
        optimizer.step()

        history.append(loss.item())
        if log_every and (step % log_every == 0 or step == 1):
            print(f"step {step:4d}/{steps}  loss {loss.item():.4f}")
    return history


@torch.no_grad()
def estimate_loss(
    model: nn.Module,
    data: torch.Tensor,
    *,
    block_size: int,
    batch_size: int = 16,
    batches: int = 20,
    seed: int = 0,
) -> float:
    """Rata-rata loss pada `batches` batch acak — estimasi mutu di data lain."""
    gen = torch.Generator().manual_seed(seed)
    was_training = model.training
    model.eval()
    losses = []
    for _ in range(batches):
        xb, yb = get_batch(data, block_size, batch_size, gen)
        _, loss = model(xb, yb)
        losses.append(loss.item())
    model.train(was_training)
    return sum(losses) / len(losses)
