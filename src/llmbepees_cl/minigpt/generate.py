"""Generasi teks: menyampel token berikutnya secara autoregresif.

Model sudah dilatih memprediksi probabilitas token berikutnya. Untuk menghasilkan
teks, kita sampel satu token, tempelkan ke konteks, lalu ulangi. `temperature`
mengatur "kenekatan" (rendah = aman/monoton, tinggi = liar/kreatif); `top_k`
membatasi pilihan ke k token teratas agar tidak memilih ekor distribusi yang aneh.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


@torch.no_grad()
def generate(
    model: nn.Module,
    idx: torch.Tensor,
    max_new_tokens: int,
    *,
    temperature: float = 1.0,
    top_k: int | None = None,
    block_size: int | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sambung `idx` ([B, T] id) dengan `max_new_tokens` token baru.

    Hanya `block_size` token terakhir yang dipakai sebagai konteks (batas posisi
    embedding). Mengembalikan tensor id dengan bentuk [B, T + max_new_tokens].
    """
    if temperature <= 0:
        raise ValueError("temperature harus > 0")
    window = block_size or getattr(model, "block_size", idx.shape[1])
    was_training = model.training
    model.eval()

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -window:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature

        if top_k is not None:
            k = min(top_k, logits.size(-1))
            top_values, _ = torch.topk(logits, k)
            logits = logits.masked_fill(logits < top_values[:, [-1]], float("-inf"))

        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1, generator=generator)
        idx = torch.cat([idx, next_id], dim=1)

    model.train(was_training)
    return idx
