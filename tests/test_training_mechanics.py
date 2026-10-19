import torch
import torch.nn.functional as F
from torch import nn

from llmbepees_cl.training import (
    autocast_context,
    clip_gradients,
    train_with_accumulation,
)


def test_clip_gradients_caps_norm():
    param = nn.Parameter(torch.zeros(4))
    param.grad = torch.tensor([3.0, 4.0, 0.0, 0.0])  # norma 5
    norm = clip_gradients([param], max_norm=1.0)
    assert abs(float(norm) - 5.0) < 1e-5
    assert abs(float(param.grad.norm()) - 1.0) < 1e-5


def test_autocast_context_is_noop_on_cpu():
    with autocast_context("cpu", enabled=True):
        x = torch.ones(2, 2)
    assert x.dtype == torch.float32


def _make_regression():
    torch.manual_seed(0)
    model = nn.Linear(4, 1)
    x = torch.randn(64, 4)
    y = x @ torch.tensor([[1.0], [-2.0], [0.5], [3.0]]) + 0.5
    gen = torch.Generator().manual_seed(0)

    def loss_fn(_micro: int) -> torch.Tensor:
        idx = torch.randint(0, 64, (16,), generator=gen)
        return F.mse_loss(model(x[idx]), y[idx])

    return model, loss_fn


def test_train_with_accumulation_reduces_loss():
    model, loss_fn = _make_regression()
    opt = torch.optim.Adam(model.parameters(), lr=0.1)
    history = train_with_accumulation(
        model, loss_fn, steps=60, optimizer=opt, accum_steps=4, max_grad_norm=1.0
    )
    assert len(history) == 60
    assert history[-1] < history[0]


def test_train_with_accumulation_requires_valid_accum():
    model, loss_fn = _make_regression()
    opt = torch.optim.Adam(model.parameters(), lr=0.1)
    try:
        train_with_accumulation(model, loss_fn, steps=1, optimizer=opt, accum_steps=0)
    except ValueError:
        return
    raise AssertionError("harus menolak accum_steps < 1")
