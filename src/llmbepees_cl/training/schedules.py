"""Jadwal learning rate: pemanasan (warmup) lalu peluruhan kosinus.

Kenapa perlu: LR konstan jarang optimal. **Warmup** mencegah langkah besar di awal
saat gradien masih liar; **peluruhan kosinus** memperkecil LR menjelang akhir agar
model "mengendap" ke minimum dengan stabil. Ini praktik standar di lab.
"""

from __future__ import annotations

import math
from collections.abc import Callable


def cosine_with_warmup(
    step: int,
    *,
    warmup_steps: int,
    max_steps: int,
    base_lr: float,
    min_lr: float = 0.0,
) -> float:
    """LR pada `step` (0-indeks): naik linier ke `base_lr`, lalu turun kosinus ke `min_lr`."""
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + coeff * (base_lr - min_lr)


def make_lr_lambda(
    *,
    warmup_steps: int,
    max_steps: int,
    min_lr_ratio: float = 0.0,
) -> Callable[[int], float]:
    """Faktor pengali LR (relatif terhadap LR awal) untuk `torch.optim.lr_scheduler.LambdaLR`."""
    return lambda step: cosine_with_warmup(
        step, warmup_steps=warmup_steps, max_steps=max_steps, base_lr=1.0, min_lr=min_lr_ratio
    )
