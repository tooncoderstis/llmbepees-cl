import pytest
import torch
from torch import nn

from llmbepees_cl.lora import (
    LoRALinear,
    apply_lora,
    mark_only_lora_trainable,
    total_parameter_count,
    trainable_parameter_count,
)
from llmbepees_cl.minigpt import MiniGPT


def test_lora_init_is_identity():
    base = nn.Linear(8, 4)
    lora = LoRALinear.from_linear(base, r=2, alpha=4)
    x = torch.randn(3, 8)
    assert torch.allclose(lora(x), base(x), atol=1e-6)


def test_lora_forward_shape():
    lora = LoRALinear(8, 4, r=2)
    assert lora(torch.randn(5, 8)).shape == (5, 4)


def test_lora_trains_only_adapters():
    lora = LoRALinear(8, 4, r=2)
    trainable = {n for n, p in lora.named_parameters() if p.requires_grad}
    assert trainable == {"lora_A", "lora_B"}


def test_lora_rejects_bad_rank():
    with pytest.raises(ValueError):
        LoRALinear(8, 4, r=0)


def test_merged_weight_matches_forward():
    torch.manual_seed(0)
    lora = LoRALinear.from_linear(nn.Linear(8, 4), r=3, alpha=6)
    lora.lora_B.data.normal_(0, 0.1)
    lora.eval()
    x = torch.randn(2, 8)
    merged = lora.merged_weight() @ x.T + lora.base.bias[:, None]
    assert torch.allclose(lora(x), merged.T, atol=1e-5)


def test_apply_lora_freezes_base_and_shrinks_trainable():
    model = MiniGPT(vocab_size=11, n_embd=16, n_head=4, n_layer=2, block_size=8)
    replaced = apply_lora(model, r=2, alpha=4)
    assert replaced == 8  # 4 proyeksi x 2 blok

    total = total_parameter_count(model)
    trainable = trainable_parameter_count(model)
    assert 0 < trainable < total * 0.2
    assert not model.blocks[0].attn.query.base.weight.requires_grad


def test_mark_only_lora_trainable():
    model = nn.Sequential(nn.Linear(4, 4), LoRALinear.from_linear(nn.Linear(4, 4), r=2))
    mark_only_lora_trainable(model)
    names = {n for n, p in model.named_parameters() if p.requires_grad}
    assert names == {"1.lora_A", "1.lora_B"}
