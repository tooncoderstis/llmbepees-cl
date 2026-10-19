"""Menerapkan LoRA ke model yang sudah ada tanpa melatih ulang seluruh bobot.

Fungsi `apply_lora` mengganti `nn.Linear` target dengan `LoRALinear` (bobot asli
dibekukan, hanya `A`/`B` yang dilatih). `mark_only_lora_trainable` membantu bila
model dimuat ulang; `trainable_parameter_count` untuk memantau efisiensi.
"""

from __future__ import annotations

from torch import nn

from llmbepees_cl.lora.layers import LoRALinear


def apply_lora(
    model: nn.Module,
    target_modules: tuple[str, ...] = ("query", "key", "value", "proj"),
    r: int = 4,
    alpha: float = 8.0,
    dropout: float = 0.0,
) -> int:
    """Ganti setiap `nn.Linear` yang namanya cocok `target_modules` dengan `LoRALinear`.

    Mencocokkan berdasarkan **nama modul** (bagian terakhir dari `named_modules`),
    mis. `attn.query`. Setelah penggantian, seluruh parameter lain dibekukan sehingga
    hanya `lora_A`/`lora_B` yang dilatih. Mengembalikan jumlah lapisan yang diganti.
    """
    replaced = 0
    for name, module in list(model.named_modules()):
        leaf = name.split(".")[-1]
        if not isinstance(module, nn.Linear) or leaf not in target_modules:
            continue
        parent = model
        parts = name.split(".")[:-1]
        for part in parts:
            parent = getattr(parent, part)
        setattr(parent, leaf, LoRALinear.from_linear(module, r=r, alpha=alpha, dropout=dropout))
        replaced += 1
    if replaced:
        mark_only_lora_trainable(model)
    return replaced


def mark_only_lora_trainable(model: nn.Module) -> None:
    """Set `requires_grad=False` untuk semua parameter kecuali `lora_A`/`lora_B`."""
    for name, param in model.named_parameters():
        param.requires_grad_("lora_A" in name or "lora_B" in name)


def trainable_parameter_count(model: nn.Module) -> int:
    """Jumlah parameter yang punya `requires_grad=True`."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def total_parameter_count(model: nn.Module) -> int:
    """Jumlah seluruh parameter (termasuk yang dibekukan)."""
    return sum(p.numel() for p in model.parameters())
