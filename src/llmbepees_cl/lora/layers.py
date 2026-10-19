"""LoRA (Low-Rank Adaptation) dari nol.

Ide inti: saat fine-tuning, jangan perbarui seluruh bobot. Bekukan bobot asli `W`
dan pelajari **perubahan berpangkat rendah** `ΔW = B·A`, dengan `A: [r, in]` dan
`B: [out, r]` serta `r ≪ min(in, out)`. Bobot efektifnya `W + (α/r)·B·A`.

Kenapa hemat: jumlah parameter yang dilatih turun dari `in×out` menjadi `r×(in+out)`.
`B` diinisialisasi nol sehingga saat mulai, `ΔW = 0` dan model berperilaku persis
seperti sebelum fine-tuning — training lalu bergerak dari titik itu.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class LoRALinear(nn.Module):
    """`nn.Linear` beku + adaptor LoRA berpangkat rendah yang bisa dilatih.

    Forward: `base(x) + scaling * (dropout(x) @ Aᵀ @ Bᵀ)`, `scaling = alpha / r`.
    Karena `B` diinisialisasi nol, output awal identik dengan `base(x)`.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 4,
        alpha: float = 8.0,
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if r <= 0:
            raise ValueError("r harus > 0")
        self.base = nn.Linear(in_features, out_features, bias=bias)
        for param in self.base.parameters():
            param.requires_grad_(False)

        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)  # mulai dari nol -> output = base
        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout)

    @classmethod
    def from_linear(
        cls, linear: nn.Linear, r: int = 4, alpha: float = 8.0, dropout: float = 0.0
    ) -> LoRALinear:
        """Bungkus `linear` yang sudah ada, salin bobotnya ke `base` yang beku."""
        layer = cls(
            linear.in_features,
            linear.out_features,
            r=r,
            alpha=alpha,
            dropout=dropout,
            bias=linear.bias is not None,
        )
        layer.base.weight.data.copy_(linear.weight.data)
        if linear.bias is not None:
            layer.base.bias.data.copy_(linear.bias.data)
        return layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.base(x)
        update = self.dropout(x) @ self.lora_A.T @ self.lora_B.T
        return base + self.scaling * update

    @torch.no_grad()
    def merged_weight(self) -> torch.Tensor:
        """Bobot efektif `W + (α/r)·B·A` — bisa dipakai untuk "melipat" LoRA ke base."""
        return self.base.weight + self.scaling * (self.lora_B @ self.lora_A)
