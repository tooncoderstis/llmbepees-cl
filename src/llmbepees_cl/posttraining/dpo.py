"""DPO (Direct Preference Optimization) dari nol.

Alih-alih melatih model reward lalu RL (RLHF/PPO), DPO langsung mengoptimalkan
kebijakan pada data preferensi (pasangan dipilih/ditolak). Kuncinya: bandingkan
**rasio log-probabilitas** policy vs reference.

Untuk tiap pasangan, dengan `β` (pengatur seberapa jauh boleh menyimpang dari
reference):

    logits = β · [ (logp_policy(y_w) − logp_policy(y_l))
                 − (logp_ref(y_w)    − logp_ref(y_l)) ]
    loss   = −logsigmoid(logits)

`y_w` = jawaban dipilih, `y_l` = ditolak. Bila policy == reference, logits = 0 dan
loss = ln 2 — titik awal yang netral. Reference **dibekukan** sebagai jangkar.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def sequence_logprobs(model: nn.Module, input_ids: torch.Tensor, prompt_len: int) -> torch.Tensor:
    """Jumlah log-probabilitas token **jawaban** (setelah `prompt_len`) — satu sekuens.

    `input_ids` adalah tensor 1D; gradien mengalir ke `model` bila tidak di-`no_grad`.
    """
    if prompt_len < 1:
        raise ValueError("prompt_len harus >= 1")
    logits, _ = model(input_ids.unsqueeze(0))
    logprobs = F.log_softmax(logits[0], dim=-1)  # [T, V]
    targets = input_ids[1:]  # [T-1]
    token_logprobs = logprobs[:-1].gather(1, targets.unsqueeze(1)).squeeze(1)
    return token_logprobs[prompt_len - 1 :].sum()


def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Hitung loss DPO dan reward (implisit) untuk jawaban dipilih/ditolak."""
    chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps)
    rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps)
    logits = chosen_rewards - rejected_rewards
    loss = -F.logsigmoid(logits).mean()
    return loss, chosen_rewards, rejected_rewards


def dpo_accuracy(chosen_rewards: torch.Tensor, rejected_rewards: torch.Tensor) -> torch.Tensor:
    """Fraksi pasangan yang reward jawaban dipilihnya lebih tinggi dari yang ditolak."""
    return (chosen_rewards > rejected_rewards).float().mean()
