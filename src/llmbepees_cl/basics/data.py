"""Data kecil untuk latihan klasifikasi — seluruhnya dibuat lokal (tanpa unduhan).

Prinsip: **deterministik** (semua fungsi menerima `seed`) dan **ramah RAM 8 GB**
(ukuran kecil, float32, tanpa dataset besar). MNIST opsional dimuat via
torchvision dengan *lazy import* supaya torchvision tidak wajib terpasang.
"""

from __future__ import annotations

import math

import torch


def make_blobs(
    n_samples: int = 300,
    n_features: int = 2,
    n_centers: int = 3,
    cluster_std: float = 0.7,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Buat gugus Gaussian ("blobs") yang terpisah dengan rapi.

    Pusat gugus diletakkan pada lingkaran berjari-jari 3 (deterministik) agar
    kelas selalu bisa dipisahkan — hasil tes tidak flaky. Mengembalikan
    `(features float32 [n_samples, n_features], labels int64 [n_samples])`
    dengan label `0..n_centers-1`, lalu diacak dengan `seed`.
    """
    if n_centers < 1:
        raise ValueError("n_centers harus >= 1")

    angles = torch.arange(n_centers, dtype=torch.float32) * (2 * math.pi / n_centers)
    centers = torch.zeros(n_centers, n_features)
    centers[:, 0] = 3.0 * torch.cos(angles)
    if n_features > 1:
        centers[:, 1] = 3.0 * torch.sin(angles)

    labels = torch.arange(n_samples) % n_centers
    gen = torch.Generator().manual_seed(seed)
    noise = torch.randn(n_samples, n_features, generator=gen) * cluster_std
    features = centers[labels] + noise

    perm = torch.randperm(n_samples, generator=gen)
    return features[perm].to(torch.float32), labels[perm].to(torch.int64)


def train_test_split(
    x: torch.Tensor,
    y: torch.Tensor,
    test_frac: float = 0.2,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Bagi data jadi latih/uji. Mengembalikan `(x_train, y_train, x_test, y_test)`."""
    n = x.shape[0]
    n_test = int(n * test_frac)
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=gen)
    test_idx, train_idx = perm[:n_test], perm[n_test:]
    return x[train_idx], y[train_idx], x[test_idx], y[test_idx]


def load_mnist(
    root: str,
    subset: int = 2000,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Muat `subset` contoh MNIST (gambar diratakan jadi vektor 784, ternormalisasi).

    Membutuhkan `torchvision` (extra `vision`) dan mengunduh dataset ke `root`.
    Arahkan `root` ke drive E:/F: agar tidak memenuhi C:. Import dilakukan di
    dalam fungsi supaya torchvision benar-benar opsional.
    """
    from torchvision import datasets, transforms

    dataset = datasets.MNIST(root=root, train=True, download=True, transform=transforms.ToTensor())
    gen = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(dataset), generator=gen)[:subset]
    x = torch.stack([dataset[int(i)][0].flatten() for i in idx]).to(torch.float32)
    y = torch.tensor([dataset[int(i)][1] for i in idx], dtype=torch.int64)
    return x, y
