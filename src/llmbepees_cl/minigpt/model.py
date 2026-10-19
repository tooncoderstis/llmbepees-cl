"""MiniGPT: blok transformer dan model bahasa kausal dari nol.

Arsitektur mengikuti GPT kecil (gaya nanoGPT): embedding token + embedding posisi,
beberapa blok transformer (attention + feed-forward), lalu kepala linear ke
kosakata. Memakai **pre-norm** (LayerNorm sebelum sub-blok) karena lebih stabil
dilatih daripada post-norm pada model kecil.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from llmbepees_cl.minigpt.attention import CausalSelfAttention


class FeedForward(nn.Module):
    """MLP per-posisi: `Linear -> GELU -> Linear`, dengan pelebaran 4x.

    Setiap token diproses sendiri-sendiri (tanpa bercampur antar posisi); tugas
    mencampur informasi antar posisi sudah diambil alih self-attention.
    """

    def __init__(self, n_embd: int, dropout: float = 0.0, expansion: int = 4) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, expansion * n_embd),
            nn.GELU(),
            nn.Linear(expansion * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):
    """Satu blok transformer: attention + feed-forward, keduanya dengan residual.

    Sisa (`x = x + sublayer(ln(x))`) membuat gradien mengalir lancar ke lapisan
    bawah, sehingga model bisa dibuat lebih dalam tanpa cepat "mati".
    """

    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffn = FeedForward(n_embd, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    """Model bahasa kausal kecil yang bisa dilatih dari nol di CPU.

    Parameter utama: `vocab_size`, `n_embd` (lebar), `n_head`, `n_layer` (kedalaman),
    dan `block_size` (panjang konteks maksimum). Forward mengembalikan
    `(logits, loss)`; `loss` hanya terisi bila `targets` diberikan.
    """

    def __init__(
        self,
        vocab_size: int,
        n_embd: int = 64,
        n_head: int = 4,
        n_layer: int = 2,
        block_size: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size

        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch, seq_len = idx.shape
        if seq_len > self.block_size:
            raise ValueError(f"seq_len {seq_len} melebihi block_size {self.block_size}")

        token = self.token_embedding(idx)  # [B, T, C]
        pos = self.position_embedding(torch.arange(seq_len, device=idx.device))  # [T, C]
        x = self.blocks(token + pos)
        x = self.ln_f(x)
        logits = self.head(x)  # [B, T, vocab_size]

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(batch * seq_len, self.vocab_size), targets.reshape(-1)
            )
        return logits, loss

    def num_parameters(self) -> int:
        """Jumlah total parameter (skalar) yang dilatih — berguna memantau ukuran model."""
        return sum(p.numel() for p in self.parameters())
