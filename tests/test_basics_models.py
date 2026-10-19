import torch

from llmbepees_cl.basics import build_mlp


def test_mlp_forward_shape():
    model = build_mlp(in_features=5, n_classes=3, hidden=(8, 4), seed=0)
    out = model(torch.randn(7, 5))
    assert out.shape == (7, 3)


def test_mlp_parameter_count_matches_layers():
    model = build_mlp(in_features=2, n_classes=3, hidden=(16,), seed=0)
    n_params = sum(p.numel() for p in model.parameters())
    # Linear(2->16): 2*16+16=48 ; Linear(16->3): 16*3+3=51
    assert n_params == 48 + 51
