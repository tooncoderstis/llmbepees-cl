"""Skrip 05 (capstone) — MLP klasifikasi lengkap: latih, evaluasi, simpan checkpoint.

Menyatukan seluruh Fase 1: data -> model -> training loop -> evaluasi -> simpan/muat.
Memakai util dari paket `llmbepees_cl.basics` (yang diuji pytest) alih-alih
menulis ulang loop di sini.

Jalankan:
    .venv\\Scripts\\python 01-pytorch-basics\\05_mlp_classifier.py
"""

from __future__ import annotations

from pathlib import Path

import torch

from llmbepees_cl.basics import (
    build_mlp,
    evaluate_accuracy,
    make_blobs,
    train_classifier,
    train_test_split,
)

# 1) Data: 3 gugus di bidang 2D, dipisah latih/uji.
x, y = make_blobs(n_samples=600, n_centers=3, cluster_std=0.6, seed=0)
x_train, y_train, x_test, y_test = train_test_split(x, y, test_frac=0.25, seed=0)

# 2) Model: MLP 2 -> 32 -> 32 -> 3.
model = build_mlp(in_features=2, n_classes=3, hidden=(32, 32), seed=0)

# 3) Latih.
history = train_classifier(model, x_train, y_train, epochs=60, lr=0.05, batch_size=32, seed=0)
for epoch in range(0, len(history), 10):
    print(f"epoch {epoch + 1:2d}  loss {history[epoch]:.4f}")
print("loss akhir  :", round(history[-1], 4))
print("akurasi uji :", round(evaluate_accuracy(model, x_test, y_test), 3))

# 4) Simpan checkpoint (state_dict = hanya bobot, bukan arsitektur).
checkpoint = Path(__file__).parent / "checkpoints" / "mlp_blobs.pt"
checkpoint.parent.mkdir(parents=True, exist_ok=True)
torch.save(model.state_dict(), checkpoint)
print("checkpoint disimpan:", checkpoint)

# 5) Muat ulang: arsitektur harus sama persis, bobot dari file.
loaded = build_mlp(in_features=2, n_classes=3, hidden=(32, 32), seed=None)
loaded.load_state_dict(torch.load(checkpoint))
print("akurasi setelah dimuat:", round(evaluate_accuracy(loaded, x_test, y_test), 3))
