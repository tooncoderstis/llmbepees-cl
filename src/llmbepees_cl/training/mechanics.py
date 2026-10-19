"""Mekanika training ala lab: akumulasi gradien, clipping, dan mixed precision.

- **Akumulasi gradien**: menjumlahkan gradien dari beberapa *micro-batch* sebelum
  `optimizer.step()`, sehingga *batch efektif* besar walau memori hanya cukup untuk
  micro-batch kecil. Trik kunci untuk RAM/VRAM terbatas (laptop 8 GB kita).
- **Gradient clipping**: membatasi norma gradien agar satu batch "buruk" tidak
  merusak bobot (meledak).
- **Mixed precision (bf16)**: berhitung di bfloat16 supaya hemat memori & cepat —
  praktis hanya di **GPU**. Di CPU kita memakai float32 (autocast dimatikan).
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import nullcontext

import torch
from torch import nn


def autocast_context(device: str | torch.device = "cpu", enabled: bool = True):
    """Konteks autocast bf16 bila `device` CUDA dan `enabled`; selain itu no-op."""
    if enabled and torch.device(device).type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def clip_gradients(parameters, max_norm: float) -> torch.Tensor:
    """Batasi norma total gradien ke `max_norm`; kembalikan norma sebelum clipping."""
    params = [p for p in parameters if p.requires_grad and p.grad is not None]
    if not params:
        return torch.tensor(0.0)
    return torch.nn.utils.clip_grad_norm_(params, max_norm)


def train_with_accumulation(
    model: nn.Module,
    loss_fn: Callable[[int], torch.Tensor],
    *,
    steps: int,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    accum_steps: int = 1,
    max_grad_norm: float | None = None,
    device: str | torch.device = "cpu",
    use_bf16: bool = False,
    log_every: int = 0,
) -> list[float]:
    """Loop training dengan akumulasi gradien + clipping opsional.

    `loss_fn(micro_index)` harus mengembalikan loss untuk **satu** micro-batch baru
    (biasanya memanggil `get_batch`). Mengembalikan riwayat loss rata-rata per langkah.
    """
    if accum_steps < 1:
        raise ValueError("accum_steps harus >= 1")
    model.train()
    history: list[float] = []
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        running = 0.0
        for micro in range(accum_steps):
            with autocast_context(device, use_bf16):
                loss = loss_fn(step * accum_steps + micro)
            (loss / accum_steps).backward()  # skala agar jumlah gradien setara rata-rata
            running += loss.item()
        if max_grad_norm is not None:
            clip_gradients(model.parameters(), max_grad_norm)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        history.append(running / accum_steps)
        if log_every and ((step + 1) % log_every == 0 or step == 0):
            print(f"step {step + 1:4d}/{steps}  loss {history[-1]:.4f}")
    return history
