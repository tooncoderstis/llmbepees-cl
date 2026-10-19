"""Skrip 03 — Merakit MiniGPT: blok transformer dan bentuk tensor.

Model bahasa kausal = embedding token + embedding posisi + tumpukan blok
transformer (attention + feed-forward) + kepala linear ke kosakata. Di sini kita
bangun modelnya, hitung jumlah parameter, dan telusuri bentuk tensor tiap tahap.

Jalankan:
    .venv\\Scripts\\python 02-mini-gpt\\03_model.py
"""

from __future__ import annotations

import torch

from llmbepees_cl.minigpt import TINY_CORPUS, CharTokenizer, MiniGPT

tok = CharTokenizer.from_text(TINY_CORPUS)
model = MiniGPT(vocab_size=tok.vocab_size, n_embd=64, n_head=4, n_layer=2, block_size=64)

print("ukuran kosakata   :", tok.vocab_size)
print("jumlah parameter  :", f"{model.num_parameters():,}")
print("konteks (block)   :", model.block_size)
print("arsitektur:\n", model)

idx = torch.randint(0, tok.vocab_size, (2, 8))  # batch 2, panjang 8
logits, loss = model(idx)
print("\nidx     :", tuple(idx.shape))
print("logits  :", tuple(logits.shape), "-> satu skor per token kosakata di tiap posisi")
print("loss    :", loss, "(None karena target tidak diberikan)")

_, loss = model(idx, idx)  # target = idx (contoh bentuk; pelatihan sesungguhnya di skrip 04)
print("loss dgn target:", round(loss.item(), 4))
print("\nCatatan: loss di sini belum bermakna; model masih acak. Melatihnya ada di 04.")
