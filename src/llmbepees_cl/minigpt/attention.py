"""Self-attention kausal — jantung transformer.

Inti idenya: setiap posisi "bertanya" (query) ke semua posisi sebelumnya dan
mengambil rata-rata nilai (value) yang bobotnya ditentukan kemiripan query-key.
Kata "kausal" berarti posisi t tidak boleh melihat masa depan (t+1, ...), supaya
model hanya belajar memprediksi token berikutnya dari konteks sebelumnya.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention (dipakai GPT).

    `n_head` kepala dihitung paralel pada dimensi `n_embd // n_head` lalu
    hasilnya disatukan dan diproyeksikan. Beberapa kepala membuat model bisa
    memperhatikan beberapa pola berbeda sekaligus.
    """

    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float = 0.0) -> None:
        super().__init__()
        if n_embd % n_head != 0:
            raise ValueError("n_embd harus habis dibagi n_head")
        self.n_head = n_head
        self.head_size = n_embd // n_head

        self.key = nn.Linear(n_embd, n_embd, bias=False)
        self.query = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(n_embd, n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

        # Masker segitiga bawah (True = boleh dilihat). Buffer ikut pindah device
        # bersama model, tapi bukan parameter yang dilatih.
        self.register_buffer(
            "mask", torch.tril(torch.ones(block_size, block_size, dtype=torch.bool))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, n_embd = x.shape

        k = self.key(x).view(batch, seq_len, self.n_head, self.head_size).transpose(1, 2)
        q = self.query(x).view(batch, seq_len, self.n_head, self.head_size).transpose(1, 2)
        v = self.value(x).view(batch, seq_len, self.n_head, self.head_size).transpose(1, 2)

        # Skor kemiripan query-key, diskalakan 1/sqrt(head_size) agar softmax stabil.
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_size)
        att = att.masked_fill(~self.mask[:seq_len, :seq_len], float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)

        out = att @ v  # [batch, n_head, seq_len, head_size]
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, n_embd)
        return self.proj(out)
