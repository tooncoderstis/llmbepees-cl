import pytest
import torch

from llmbepees_cl.minigpt import MiniGPT


def make_model(**kwargs):
    defaults = {"vocab_size": 11, "n_embd": 16, "n_head": 4, "n_layer": 2, "block_size": 8}
    return MiniGPT(**{**defaults, **kwargs})


def test_forward_shapes_without_targets():
    model = make_model()
    logits, loss = model(torch.randint(0, 11, (2, 8)))
    assert logits.shape == (2, 8, 11)
    assert loss is None


def test_forward_loss_is_scalar():
    model = make_model()
    x = torch.randint(0, 11, (3, 8))
    logits, loss = model(x, x)
    assert logits.shape == (3, 8, 11)
    assert loss.ndim == 0
    assert loss.item() > 0


def test_block_size_is_enforced():
    model = make_model(block_size=4)
    with pytest.raises(ValueError):
        model(torch.randint(0, 11, (1, 5)))


def test_parameter_count_positive():
    model = make_model(n_layer=3)
    assert model.num_parameters() > 0
