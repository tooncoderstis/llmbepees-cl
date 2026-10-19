"""Model jaringan saraf sederhana (MLP) untuk latihan klasifikasi."""

from __future__ import annotations

import torch
from torch import nn


class MLP(nn.Module):
    """Klasifier feed-forward: `Linear -> ReLU` (diulang `hidden`) lalu `Linear` terakhir.

    Mengembalikan **logits** (belum softmax) karena `nn.CrossEntropyLoss`
    sudah menghitung softmax di dalamnya — ini lebih stabil secara numerik.
    """

    def __init__(
        self,
        in_features: int,
        n_classes: int,
        hidden: tuple[int, ...] = (32,),
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_features
        for width in hidden:
            layers.append(nn.Linear(prev, width))
            layers.append(nn.ReLU())
            prev = width
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_mlp(
    in_features: int,
    n_classes: int,
    hidden: tuple[int, ...] = (32,),
    seed: int | None = 0,
) -> MLP:
    """Bangun `MLP`; bila `seed` diisi, inisialisasi bobot jadi dapat direproduksi."""
    if seed is not None:
        torch.manual_seed(seed)
    return MLP(in_features, n_classes, hidden)
