import torch

from llmbepees_cl.basics import numerical_grad


def test_numerical_grad_matches_autograd():
    x = torch.tensor([1.5, -2.0], dtype=torch.float32)

    def f(t: torch.Tensor) -> torch.Tensor:
        return (t**3).sum() + 2 * t[0]

    numeric = numerical_grad(f, x.clone(), eps=1e-2)

    x_grad = x.clone().requires_grad_(True)
    f(x_grad).backward()

    # d/dt (t^3) = 3t^2 ; turunan elemen pertama +2 dari suku 2*t[0]
    assert torch.allclose(numeric, x_grad.grad, atol=1e-2)
