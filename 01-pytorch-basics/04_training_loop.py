"""Skrip 04 — Anatomi training loop: dari data mentah sampai bobot terlatih.

Lima langkah inti setiap iterasi (hafalkan urutannya):
    zero_grad -> forward -> loss -> backward -> step

Jalankan:
    .venv\\Scripts\\python 01-pytorch-basics\\04_training_loop.py
"""

from __future__ import annotations

import torch
from torch import nn

from llmbepees_cl.basics import make_blobs, train_test_split

x, y = make_blobs(n_samples=300, n_centers=3, seed=0)
x_train, y_train, x_test, y_test = train_test_split(x, y, seed=0)

# Model dibangun langsung dengan nn.Sequential (versi ringkas dari nn.Module).
model = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 3))
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

batch_size = 32
for epoch in range(1, 31):
    perm = torch.randperm(x_train.shape[0])  # acak urutan tiap epoch
    for start in range(0, x_train.shape[0], batch_size):
        idx = perm[start : start + batch_size]
        xb, yb = x_train[idx], y_train[idx]

        optimizer.zero_grad()  # (1) reset gradien lama
        logits = model(xb)  # (2) forward
        loss = loss_fn(logits, yb)  # (3) hitung loss
        loss.backward()  # (4) backward -> isi .grad
        optimizer.step()  # (5) perbarui bobot

    if epoch % 5 == 0:
        with torch.no_grad():
            acc = (model(x_test).argmax(dim=1) == y_test).float().mean().item()
        print(f"epoch {epoch:2d}  loss {loss.item():.4f}  akurasi_uji {acc:.3f}")
