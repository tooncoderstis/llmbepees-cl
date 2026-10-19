"""Loop training DPO minimalis: policy dilatih, reference dibekukan.

Mirip training biasa (`zero_grad → forward → loss → backward → step`), tetapi loss
membandingkan policy vs reference pada tiap pasangan preferensi. Untuk tiap langkah
kita ambil beberapa pasangan (`batch_size`) dan akumulasi gradiennya.
"""

from __future__ import annotations

import torch
from torch import nn

from llmbepees_cl.posttraining.data import PreferencePair
from llmbepees_cl.posttraining.dpo import dpo_accuracy, dpo_loss, sequence_logprobs


def encode_prompt_response(tokenizer, prompt: str, response: str) -> tuple[torch.Tensor, int]:
    """Encode `prompt + response`; kembalikan id dan panjang prompt (token)."""
    prompt_ids = tokenizer.encode(prompt)
    full_ids = tokenizer.encode(prompt + response)
    return torch.tensor(full_ids, dtype=torch.long), len(prompt_ids)


def train_dpo(
    policy: nn.Module,
    reference: nn.Module,
    pairs: list[PreferencePair],
    tokenizer,
    *,
    steps: int = 200,
    batch_size: int = 4,
    beta: float = 0.1,
    lr: float = 1e-4,
    seed: int | None = 0,
    log_every: int = 0,
) -> list[float]:
    """Latih `policy` dengan DPO; `reference` dibekukan. Mengembalikan riwayat loss."""
    if not pairs:
        raise ValueError("butuh minimal satu pasangan preferensi")
    if seed is not None:
        torch.manual_seed(seed)
    for param in reference.parameters():
        param.requires_grad_(False)
    reference.eval()

    optimizer = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=lr)
    gen = torch.Generator().manual_seed(seed if seed is not None else 0)
    history: list[float] = []

    policy.train()
    for step in range(1, steps + 1):
        indices = torch.randint(0, len(pairs), (batch_size,), generator=gen)
        optimizer.zero_grad(set_to_none=True)
        running = 0.0
        for i in indices:
            pair = pairs[int(i)]

            chosen_ids, prompt_len = encode_prompt_response(tokenizer, pair.prompt, pair.chosen)
            rejected_ids, _ = encode_prompt_response(tokenizer, pair.prompt, pair.rejected)

            policy_chosen = sequence_logprobs(policy, chosen_ids, prompt_len)
            policy_rejected = sequence_logprobs(policy, rejected_ids, prompt_len)
            with torch.no_grad():
                ref_chosen = sequence_logprobs(reference, chosen_ids, prompt_len)
                ref_rejected = sequence_logprobs(reference, rejected_ids, prompt_len)

            loss, _, _ = dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta)
            (loss / batch_size).backward()
            running += loss.item()

        optimizer.step()
        history.append(running / batch_size)
        if log_every and (step % log_every == 0 or step == 1):
            print(f"step {step:4d}/{steps}  loss {history[-1]:.4f}")
    return history


@torch.no_grad()
def evaluate_dpo(
    policy: nn.Module,
    reference: nn.Module,
    pairs: list[PreferencePair],
    tokenizer,
    beta: float = 0.1,
) -> dict[str, float]:
    """Hitung loss rata-rata dan akurasi preferensi pada seluruh `pairs`."""
    policy.eval()
    losses, accs = [], []
    for pair in pairs:
        chosen_ids, prompt_len = encode_prompt_response(tokenizer, pair.prompt, pair.chosen)
        rejected_ids, _ = encode_prompt_response(tokenizer, pair.prompt, pair.rejected)
        pc = sequence_logprobs(policy, chosen_ids, prompt_len)
        pr = sequence_logprobs(policy, rejected_ids, prompt_len)
        rc = sequence_logprobs(reference, chosen_ids, prompt_len)
        rr = sequence_logprobs(reference, rejected_ids, prompt_len)
        loss, cr, rj = dpo_loss(pc, pr, rc, rr, beta)
        losses.append(loss.item())
        accs.append(dpo_accuracy(cr.unsqueeze(0), rj.unsqueeze(0)).item())
    return {"loss": sum(losses) / len(losses), "accuracy": sum(accs) / len(accs)}
