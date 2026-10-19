"""Skrip 06 (opsional) — MNIST subset kecil lewat torchvision.

Menguji keterampilan yang sama pada data nyata (gambar 28x28 digit tulisan tangan).
Dataset diunduh ke `01-pytorch-basics/data/mnist/` (di-ignore git) dengan `subset`
kecil supaya tetap ramah CPU/RAM.

Butuh ekstra `vision` (torchvision). Jalankan:
    .venv\\Scripts\\python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    .venv\\Scripts\\python 01-pytorch-basics\\06_mnist_optional.py
"""

from __future__ import annotations

from pathlib import Path

from llmbepees_cl.basics import (
    build_mlp,
    evaluate_accuracy,
    load_mnist,
    train_classifier,
    train_test_split,
)

root = Path(__file__).parent / "data" / "mnist"
print("mengunduh/memuat MNIST ke:", root, "(pertama kali agak lama)...")
x, y = load_mnist(str(root), subset=4000, seed=0)
x_train, y_train, x_test, y_test = train_test_split(x, y, test_frac=0.2, seed=0)
print("bentuk data:", x.shape)

model = build_mlp(in_features=784, n_classes=10, hidden=(64,), seed=0)
history = train_classifier(model, x_train, y_train, epochs=5, lr=0.01, batch_size=64, seed=0)
print("loss per epoch:", [round(v, 3) for v in history])
print("akurasi uji:", round(evaluate_accuracy(model, x_test, y_test), 3))
