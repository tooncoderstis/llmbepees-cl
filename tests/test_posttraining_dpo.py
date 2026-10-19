import copy
import math

import torch

from llmbepees_cl.minigpt import TINY_CORPUS, CharTokenizer, MiniGPT
from llmbepees_cl.posttraining import (
    PREFERENCE_PAIRS,
    dpo_accuracy,
    dpo_loss,
    encode_prompt_response,
    evaluate_dpo,
    sequence_logprobs,
    train_dpo,
)


def _tokenizer():
    text = TINY_CORPUS + "".join(p.prompt + p.chosen + p.rejected for p in PREFERENCE_PAIRS)
    return CharTokenizer.from_text(text)


def _models(seed: int = 0):
    tok = _tokenizer()
    policy = MiniGPT(vocab_size=tok.vocab_size, n_embd=32, n_head=2, n_layer=1, block_size=64)
    reference = copy.deepcopy(policy)
    return policy, reference, tok


def test_dpo_loss_is_ln2_when_policy_equals_reference():
    zeros = torch.zeros(4)
    loss, chosen, rejected = dpo_loss(zeros, zeros, zeros, zeros, beta=0.1)
    assert abs(loss.item() - math.log(2)) < 1e-6
    assert torch.allclose(chosen, torch.zeros(4))
    assert torch.allclose(rejected, torch.zeros(4))


def test_dpo_accuracy():
    chosen = torch.tensor([1.0, 2.0, -1.0])
    rejected = torch.tensor([0.0, 3.0, -2.0])
    assert abs(dpo_accuracy(chosen, rejected).item() - 2 / 3) < 1e-6


def test_sequence_logprobs_is_negative_and_trainable():
    policy, _, tok = _models()
    ids, prompt_len = encode_prompt_response(
        tok, PREFERENCE_PAIRS[0].prompt, PREFERENCE_PAIRS[0].chosen
    )
    logp = sequence_logprobs(policy, ids, prompt_len)
    assert logp.ndim == 0
    assert logp.item() < 0
    assert logp.requires_grad


def test_encode_prompt_response_prompt_len():
    tok = _tokenizer()
    ids, prompt_len = encode_prompt_response(tok, "T: inflasi\nJ:", " naiknya harga.\n")
    assert prompt_len == len(tok.encode("T: inflasi\nJ:"))
    assert ids.shape[0] == prompt_len + len(tok.encode(" naiknya harga.\n"))


def test_reference_is_frozen():
    policy, reference, tok = _models()
    before = [p.detach().clone() for p in reference.parameters()]
    train_dpo(policy, reference, PREFERENCE_PAIRS, tok, steps=20, batch_size=2, lr=1e-4)
    after = list(reference.parameters())
    assert all(torch.equal(b, a) for b, a in zip(before, after, strict=True))
    assert all(not p.requires_grad for p in reference.parameters())


def test_train_dpo_reduces_loss_and_improves_accuracy():
    policy, reference, tok = _models()
    before = evaluate_dpo(policy, reference, PREFERENCE_PAIRS, tok)
    history = train_dpo(policy, reference, PREFERENCE_PAIRS, tok, steps=200, batch_size=4, lr=1e-4)
    after = evaluate_dpo(policy, reference, PREFERENCE_PAIRS, tok)

    assert abs(before["loss"] - math.log(2)) < 1e-4
    assert history[-1] < before["loss"]
    assert after["loss"] < before["loss"]
    assert after["accuracy"] > before["accuracy"]
