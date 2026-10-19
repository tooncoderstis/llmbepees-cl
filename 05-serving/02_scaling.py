"""Skrip 02 — Scaling law kecil: loss vs ukuran model.

Kita pakai **tugas sintetis dengan data tak terbatas**: deret token yang token
berikutnya ditentukan fungsi dari dua token sebelumnya, `f(a, b) = (a·5 + b·3) mod V`.
Model harus belajar fungsi itu. Karena data selalu baru, loss validasi mengukur
**generalisasi**, bukan hafalan — sehingga menambah kapasitas model menurunkan loss.
Lalu kita pasang hukum pangkat `L = a·N^b` (N = jumlah parameter).

Kenapa sintetis? Korpus teks kecil membuat model besar **overfit**, sehingga loss
validasi justru naik — pelajaran: scaling law butuh **data & komputasi cukup**.
Sumber: Kaplan dkk. 2020; Hoffmann dkk. (Chinchilla) 2022.

Jalankan:
    .venv\\Scripts\\python 05-serving\\02_scaling.py
"""

from __future__ import annotations

import torch

from llmbepees_cl.minigpt import MiniGPT
from llmbepees_cl.training import fit_power_law, train_with_accumulation

VOCAB = 12
BLOCK = 8


def make_batch(gen: torch.Generator, batch: int, block: int = BLOCK):
    """Buat batch baru: token ke-i ditentukan oleh dua token sebelumnya (deterministik)."""
    a = torch.randint(0, VOCAB, (batch,), generator=gen)
    b = torch.randint(0, VOCAB, (batch,), generator=gen)
    seq = [a, b]
    for _ in range(block):
        seq.append((seq[-1] * 5 + seq[-2] * 3) % VOCAB)
    seq = torch.stack(seq, dim=1)  # [batch, block+2]
    return seq[:, :block], seq[:, 1 : block + 1]


def val_loss(model: MiniGPT, batches: int = 20, seed: int = 123) -> float:
    model.eval()
    gen = torch.Generator().manual_seed(seed)
    total = 0.0
    with torch.no_grad():
        for _ in range(batches):
            x, y = make_batch(gen, 64)
            _, loss = model(x, y)
            total += loss.item()
    model.train()
    return total / batches


def make_loss(model: MiniGPT, gen: torch.Generator):
    def loss_fn(_micro: int) -> torch.Tensor:
        x, y = make_batch(gen, 32)
        _, loss = model(x, y)
        return loss

    return loss_fn


widths = [16, 32, 64]
params: list[int] = []
losses: list[float] = []

print(f"{'n_embd':>7} {'params':>9} {'loss val':>9}")
for width in widths:
    model = MiniGPT(vocab_size=VOCAB, n_embd=width, n_head=4, n_layer=2, block_size=BLOCK)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    gen = torch.Generator().manual_seed(0)
    train_with_accumulation(model, make_loss(model, gen), steps=300, optimizer=opt)
    params.append(model.num_parameters())
    losses.append(val_loss(model))
    print(f"{width:>7} {model.num_parameters():>9,} {losses[-1]:>9.4f}")

a, b = fit_power_law(params, losses)
print(f"\nhukum pangkat: loss ~ {a:.3f} * N^({b:.3f})")
print("eksponen b negatif = menambah parameter menurunkan loss (arah scaling law).")
