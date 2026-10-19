"""Skrip 05 — Menghasilkan teks dari MiniGPT terlatih.

Memuat checkpoint hasil `04_train.py`, memberi beberapa "prompt", lalu menyampel
token berikutnya secara berulang. Bandingkan efek `temperature` (rendah = aman,
tinggi = liar) dan `top_k` (membatasi pilihan ke k token teratas).

Jalankan (setelah 04):
    .venv\\Scripts\\python 02-mini-gpt\\05_generate.py
"""

from __future__ import annotations

from pathlib import Path

import torch

from llmbepees_cl.minigpt import CharTokenizer, MiniGPT, generate

ckpt_path = Path(__file__).parent / "checkpoints" / "minigpt.pt"
if not ckpt_path.exists():
    raise SystemExit("checkpoint belum ada — jalankan dulu: 02-mini-gpt\\04_train.py")

ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
tok = CharTokenizer(ckpt["itos"])
model = MiniGPT(vocab_size=tok.vocab_size, **ckpt["config"])
model.load_state_dict(ckpt["state_dict"])
model.eval()


def sample(prompt: str, n_new: int, temperature: float, top_k: int | None) -> str:
    idx = torch.tensor([tok.encode(prompt)], dtype=torch.long)
    gen = torch.Generator().manual_seed(0)
    out = generate(
        model, idx, max_new_tokens=n_new, temperature=temperature, top_k=top_k, generator=gen
    )
    return tok.decode(out[0].tolist())


prompt = "Data statistik adalah "
print(f"prompt: {prompt!r}\n")
for temperature in (0.5, 0.8, 1.2):
    hasil = sample(prompt, n_new=120, temperature=temperature, top_k=None)
    print(f"[temperature={temperature}] {hasil}\n")

print("[top_k=5, temperature=0.8]")
print(sample(prompt, n_new=120, temperature=0.8, top_k=5))
