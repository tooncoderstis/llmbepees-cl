"""Skrip 04 — Melatih MiniGPT pada korpus kecil.

Tugasnya sederhana: untuk setiap posisi, tebak token berikutnya. Loss cross-entropy
turun seiring model belajar pola statistik teks (di sini: kosakata/definisi seputar
statistik dari `TINY_CORPUS`). Bobot disimpan ke `checkpoints/minigpt.pt`.

Jalankan:
    .venv\\Scripts\\python 02-mini-gpt\\04_train.py
"""

from __future__ import annotations

from pathlib import Path

import torch

from llmbepees_cl.minigpt import (
    TINY_CORPUS,
    CharTokenizer,
    MiniGPT,
    encode_corpus,
    estimate_loss,
    split_data,
    train_language_model,
)

config = {"n_embd": 64, "n_head": 4, "n_layer": 2, "block_size": 64}
steps = 800

tok = CharTokenizer.from_text(TINY_CORPUS)
data = encode_corpus(TINY_CORPUS, tok)
train_data, val_data = split_data(data, val_frac=0.1)
print(f"token total: {data.shape[0]} (latih {train_data.shape[0]}, val {val_data.shape[0]})")

model = MiniGPT(vocab_size=tok.vocab_size, **config)
print(f"parameter: {model.num_parameters():,}")

history = train_language_model(
    model,
    train_data,
    steps=steps,
    batch_size=16,
    block_size=config["block_size"],
    lr=3e-3,
    seed=0,
    log_every=100,
)

val_loss = estimate_loss(model, val_data, block_size=config["block_size"], batch_size=16)
print(f"\nloss latih awal : {history[0]:.4f}")
print(f"loss latih akhir: {history[-1]:.4f}")
print(
    f"loss validasi   : {val_loss:.4f}  (acuan tebakan acak = ln({tok.vocab_size}) ~ "
    f"{torch.log(torch.tensor(float(tok.vocab_size))).item():.4f})"
)

ckpt_dir = Path(__file__).parent / "checkpoints"
ckpt_dir.mkdir(exist_ok=True)
ckpt_path = ckpt_dir / "minigpt.pt"
torch.save(
    {"config": config, "itos": tok.itos, "state_dict": model.state_dict()},
    ckpt_path,
)
print("\ncheckpoint disimpan:", ckpt_path)
