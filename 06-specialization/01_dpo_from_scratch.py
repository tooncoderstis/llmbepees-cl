"""Skrip 01 — DPO dari nol pada MiniGPT: melatih preferensi, bukan sekadar menebak.

Kita mulai dari model (acak) + salinannya sebagai **reference** (dibekukan). DPO lalu
"mendorong" policy agar memberi probabilitas lebih tinggi pada jawaban yang dipilih
(memberi definisi) dibanding yang ditolak (menolak menjawab), dengan reference sebagai
jangkar. Di awal, policy == reference sehingga loss = ln 2 ≈ 0.693 (netral).

Catatan: ini MiniGPT karakter yang sangat kecil — tujuannya memahami **mekanika DPO**,
bukan memproduksi bahasa mulus (versi model nyata ada di skrip Colab).

Jalankan:
    .venv\\Scripts\\python 06-specialization\\01_dpo_from_scratch.py
"""

from __future__ import annotations

import copy
import math

from llmbepees_cl.minigpt import TINY_CORPUS, CharTokenizer, MiniGPT
from llmbepees_cl.posttraining import (
    PREFERENCE_PAIRS,
    dpo_loss,
    encode_prompt_response,
    evaluate_dpo,
    sequence_logprobs,
    train_dpo,
)

text = TINY_CORPUS + "".join(p.prompt + p.chosen + p.rejected for p in PREFERENCE_PAIRS)
tok = CharTokenizer.from_text(text)

policy = MiniGPT(vocab_size=tok.vocab_size, n_embd=64, n_head=4, n_layer=2, block_size=64)
reference = copy.deepcopy(policy)  # jangkar, dibekukan

print("loss netral (ln 2) =", round(math.log(2), 4))
print(
    "sebelum DPO:",
    {k: round(v, 4) for k, v in evaluate_dpo(policy, reference, PREFERENCE_PAIRS, tok).items()},
)

history = train_dpo(
    policy,
    reference,
    PREFERENCE_PAIRS,
    tok,
    steps=300,
    batch_size=4,
    beta=0.1,
    lr=1e-4,
    log_every=100,
)
print(
    "sesudah DPO:",
    {k: round(v, 4) for k, v in evaluate_dpo(policy, reference, PREFERENCE_PAIRS, tok).items()},
)
print(f"loss: {history[0]:.4f} -> {history[-1]:.4f}")


def rewards(model: MiniGPT, pair) -> tuple[float, float]:
    """Reward implisit (beta·(logp_policy − logp_ref)) untuk jawaban dipilih vs ditolak."""
    chosen_ids, prompt_len = encode_prompt_response(tok, pair.prompt, pair.chosen)
    rejected_ids, _ = encode_prompt_response(tok, pair.prompt, pair.rejected)
    pc = sequence_logprobs(model, chosen_ids, prompt_len)
    pr = sequence_logprobs(model, rejected_ids, prompt_len)
    rc = sequence_logprobs(reference, chosen_ids, prompt_len)
    rr = sequence_logprobs(reference, rejected_ids, prompt_len)
    _, cr, rj = dpo_loss(pc, pr, rc, rr, beta=0.1)
    return cr.item(), rj.item()


pair = PREFERENCE_PAIRS[0]  # "T: inflasi\nJ:"
print(f"\npasangan: {pair.prompt!r}")
print(f"  dipilih : {pair.chosen!r}")
print(f"  ditolak : {pair.rejected!r}")
print(
    "reward (dipilih, ditolak) sebelum DPO:", tuple(round(v, 3) for v in rewards(reference, pair))
)
print("reward (dipilih, ditolak)  sesudah DPO:", tuple(round(v, 3) for v in rewards(policy, pair)))
print("\nPelajaran: DPO menaikkan reward jawaban dipilih relatif terhadap yang ditolak,")
print("tanpa menyentuh reference; loss turun dari ln 2 dan akurasi preferensi naik.")
