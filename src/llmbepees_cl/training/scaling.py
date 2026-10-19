"""Scaling laws: hubungan hukum pangkat antara ukuran model/data dan loss.

Hasil empiris (Kaplan 2020; Chinchilla 2022): loss menurun sebagai hukum pangkat
terhadap jumlah parameter (N), data (D), dan komputasi (C) — `L = a · X^(-b)`.
Di sini kita sediakan pemasangan (fitting) sederhana dari pasangan (X, loss) supaya
eksperimen kecil di laptop bisa diringkas menjadi satu eksponen.
"""

from __future__ import annotations

import numpy as np


def fit_power_law(xs, ys) -> tuple[float, float]:
    """Pasang `y = a · x^b` via regresi linier pada skala log-log.

    Mengembalikan `(a, b)`. `b` negatif berarti loss turun saat `x` naik.
    """
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    if xs.size < 2 or np.any(xs <= 0) or np.any(ys <= 0):
        raise ValueError("butuh minimal 2 titik positif")
    slope, intercept = np.polyfit(np.log(xs), np.log(ys), 1)
    return float(np.exp(intercept)), float(slope)


def predict(law: tuple[float, float], x: float) -> float:
    """Prediksi loss `y` untuk `x` memakai hukum pangkat `(a, b)`."""
    a, b = law
    return a * (x**b)
