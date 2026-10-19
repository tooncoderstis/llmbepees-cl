"""Skrip 01 — Mekanika training: schedule LR, akumulasi gradien, clipping.

Memperlihatkan tiga alat yang dipakai di lab untuk melatih dengan stabil dan hemat
memori: (1) jadwal LR warmup+cosine, (2) akumulasi gradien (batch efektif besar
walau micro-batch kecil), (3) gradient clipping. Dijalankan di MiniGPT (CPU).

Jalankan:
    .venv\\Scripts\\python 05-serving\\01_train_mechanics.py
"""

from __future__ import annotations

import torch

from llmbepees_cl.minigpt import TINY_CORPUS, CharTokenizer, MiniGPT, encode_corpus, get_batch
from llmbepees_cl.training import (
    cosine_with_warmup,
    make_lr_lambda,
    train_with_accumulation,
)

print("Contoh jadwal LR (base=1.0, warmup=20, max=200):")
for step in (0, 5, 19, 100, 199):
    lr = cosine_with_warmup(step, warmup_steps=20, max_steps=200, base_lr=1.0)
    print(f"  step {step:3d}: lr = {lr:.4f}")

tok = CharTokenizer.from_text(TINY_CORPUS)
data = encode_corpus(TINY_CORPUS, tok)
model = MiniGPT(vocab_size=tok.vocab_size, n_embd=64, n_head=4, n_layer=2, block_size=64)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer, make_lr_lambda(warmup_steps=20, max_steps=200)
)
gen = torch.Generator().manual_seed(0)


def loss_fn(_micro: int) -> torch.Tensor:
    xb, yb = get_batch(data, 64, 8, gen)  # micro-batch 8
    _, loss = model(xb, yb)
    return loss


print("\nMelatih: micro-batch 8 x akumulasi 4 = batch efektif 32")
history = train_with_accumulation(
    model,
    loss_fn,
    steps=200,
    optimizer=optimizer,
    scheduler=scheduler,
    accum_steps=4,
    max_grad_norm=1.0,
    log_every=50,
)
print(
    f"\nloss {history[0]:.3f} -> {history[-1]:.3f} | lr akhir {optimizer.param_groups[0]['lr']:.5f}"
)
print("\nPelajaran: akumulasi memberi batch efektif besar dengan memori micro-batch;")
print("warmup mencegah langkah besar di awal; clipping menahan gradien meledak.")
