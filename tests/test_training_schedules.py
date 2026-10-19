from llmbepees_cl.training import cosine_with_warmup, make_lr_lambda


def test_warmup_reaches_base_lr():
    lr = cosine_with_warmup(4, warmup_steps=5, max_steps=100, base_lr=1.0)
    assert abs(lr - 1.0) < 1e-9  # step ke-5 (indeks 4) sudah penuh


def test_warmup_starts_small():
    lr = cosine_with_warmup(0, warmup_steps=5, max_steps=100, base_lr=1.0)
    assert abs(lr - 0.2) < 1e-9


def test_cosine_decays_to_min():
    lr = cosine_with_warmup(100, warmup_steps=5, max_steps=100, base_lr=1.0, min_lr=0.1)
    assert abs(lr - 0.1) < 1e-9


def test_cosine_is_decreasing_after_warmup():
    values = [
        cosine_with_warmup(s, warmup_steps=5, max_steps=100, base_lr=1.0) for s in range(5, 100)
    ]
    assert all(a >= b for a, b in zip(values, values[1:], strict=False))


def test_lr_lambda_is_relative_multiplier():
    fn = make_lr_lambda(warmup_steps=5, max_steps=100, min_lr_ratio=0.0)
    assert abs(fn(4) - 1.0) < 1e-9
    assert fn(100) == 0.0
