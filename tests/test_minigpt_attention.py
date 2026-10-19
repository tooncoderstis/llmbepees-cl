import pytest
import torch

from llmbepees_cl.minigpt import CausalSelfAttention


def test_attention_preserves_shape():
    attn = CausalSelfAttention(n_embd=16, n_head=4, block_size=8)
    x = torch.randn(2, 5, 16)
    assert attn(x).shape == (2, 5, 16)


def test_attention_is_causal():
    torch.manual_seed(0)
    attn = CausalSelfAttention(n_embd=16, n_head=4, block_size=8).eval()
    x = torch.randn(1, 5, 16)
    out = attn(x)

    x2 = x.clone()
    x2[:, 3:] = torch.randn(1, 2, 16)
    out2 = attn(x2)

    # Posisi 0..2 tidak boleh terpengaruh token masa depan (posisi 3..4).
    assert torch.allclose(out[:, :3], out2[:, :3], atol=1e-5)


def test_attention_requires_divisible_heads():
    with pytest.raises(ValueError):
        CausalSelfAttention(n_embd=10, n_head=4, block_size=8)
