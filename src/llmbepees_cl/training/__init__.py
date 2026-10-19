"""Mekanika training & scaling (Fase 5): jadwal LR, akumulasi gradien, mixed precision, scaling law.

Reusable dan diuji di `tests/test_training_*.py`. Skrip belajar ada di `05-serving/`.
"""

from llmbepees_cl.training.mechanics import (
    autocast_context,
    clip_gradients,
    train_with_accumulation,
)
from llmbepees_cl.training.scaling import fit_power_law, predict
from llmbepees_cl.training.schedules import cosine_with_warmup, make_lr_lambda

__all__ = [
    "autocast_context",
    "clip_gradients",
    "cosine_with_warmup",
    "fit_power_law",
    "make_lr_lambda",
    "predict",
    "train_with_accumulation",
]
