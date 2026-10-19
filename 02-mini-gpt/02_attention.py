"""Skrip 02 — Self-attention kausal: siapa memperhatikan siapa.

Attention menghitung skor kemiripan antara query (tiap posisi) dan key (posisi
lain), lalu memakai skor itu sebagai bobot rata-rata atas value. Masker kausal
memastikan posisi t hanya melihat 0..t (tidak boleh mengintip masa depan).

Jalankan:
    .venv\\Scripts\\python 02-mini-gpt\\02_attention.py
"""

from __future__ import annotations

import torch

from llmbepees_cl.minigpt import CausalSelfAttention

torch.manual_seed(0)
block_size = 6
attn = CausalSelfAttention(n_embd=16, n_head=4, block_size=block_size).eval()

x = torch.randn(1, block_size, 16)
out = attn(x)
print("bentuk input :", tuple(x.shape))
print("bentuk output:", tuple(out.shape), "(dimensi tetap sama)")

# Tampilkan matriks perhatian satu kepala supaya struktur kausal terlihat.
seq_len = block_size
with torch.no_grad():
    q = attn.query(x).view(1, seq_len, attn.n_head, attn.head_size).transpose(1, 2)
    k = attn.key(x).view(1, seq_len, attn.n_head, attn.head_size).transpose(1, 2)
    att = (q @ k.transpose(-2, -1)) / (attn.head_size**0.5)
    att = att.masked_fill(~attn.mask[:seq_len, :seq_len], float("-inf"))
    att = torch.softmax(att, dim=-1)
print("\nMatriks perhatian (kepala 0) — baris=posisi query, kolom=posisi key:")
print(torch.round(att[0, 0], decimals=2))

# Uji kausal: ganti token masa depan (posisi 4-5), posisi 0-3 harus tetap sama.
x2 = x.clone()
x2[:, 4:] = torch.randn(1, 2, 16)
out2 = attn(x2)
print("\nselisih posisi 0-3 setelah masa depan diubah (harus ~0):")
print((out[:, :4] - out2[:, :4]).abs().max().item())
print("selisih posisi 4-5 (boleh != 0):", (out[:, 4:] - out2[:, 4:]).abs().max().item())
