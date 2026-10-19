"""Skrip 01 — LoRA dari nol: fine-tuning hemat parameter.

Kita punya jaringan dasar yang sudah dilatih untuk satu tugas (A). Lalu kita
"fine-tune" ke tugas sempit lain (B) **tanpa** mengubah bobot dasar: hanya adaptor
LoRA berpangkat rendah (A·B) yang dilatih. Perhatikan berapa sedikit parameter yang
diperbarui, dan bahwa output sebelum pelatihan LoRA identik dengan model dasar.

Jalankan:
    .venv\\Scripts\\python 03-finetune-lora\\01_lora_from_scratch.py
"""

from __future__ import annotations

import torch
from torch import nn

from llmbepees_cl.lora import apply_lora, total_parameter_count, trainable_parameter_count

torch.manual_seed(0)
n_in, n_out = 8, 4

# --- Tugas A: jaringan dasar dilatih memetakan x -> W_A x ---------------------
base = nn.Sequential(nn.Linear(n_in, 16), nn.ReLU(), nn.Linear(16, n_out))
w_a = torch.randn(n_out, n_in)
x = torch.randn(64, n_in)
y_a = x @ w_a.T

opt = torch.optim.Adam(base.parameters(), lr=1e-2)
for _ in range(300):
    opt.zero_grad()
    loss = nn.functional.mse_loss(base(x), y_a)
    loss.backward()
    opt.step()
print(f"[A] loss setelah latih dasar: {loss.item():.4f}")

# --- Fine-tune ke tugas B dengan LoRA -----------------------------------------
w_b = torch.randn(n_out, n_in)  # tugas baru: x -> W_B x
y_b = x @ w_b.T

replaced = apply_lora(base, target_modules=("0", "2"), r=2, alpha=4)
before = nn.functional.mse_loss(base(x), y_b).item()

base_weight_before = base[0].base.weight.detach().clone()
params = [p for p in base.parameters() if p.requires_grad]
opt = torch.optim.Adam(params, lr=1e-2)
for _ in range(400):
    opt.zero_grad()
    loss = nn.functional.mse_loss(base(x), y_b)
    loss.backward()
    opt.step()

after = nn.functional.mse_loss(base(x), y_b).item()
base_unchanged = torch.equal(base[0].base.weight, base_weight_before)

total = total_parameter_count(base)
trainable = trainable_parameter_count(base)

print(f"lapisan diganti LoRA : {replaced}")
print(f"total parameter      : {total}")
print(f"parameter dilatih    : {trainable}  ({100 * trainable / total:.1f}% dari total)")
print(f"loss tugas B sebelum : {before:.4f}")
print(f"loss tugas B sesudah : {after:.4f}")
print(f"bobot dasar tak berubah: {base_unchanged}")
print("\nPelajaran: LoRA hanya melatih matriks berpangkat rendah (A·B),")
print("namun cukup untuk berpindah tugas — dan bobot dasar tetap utuh.")
