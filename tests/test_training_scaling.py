import pytest

from llmbepees_cl.training import fit_power_law, predict


def test_fit_power_law_recovers_exponent():
    xs = [1, 2, 4, 8, 16]
    ys = [2.0 * (x**-0.5) for x in xs]
    a, b = fit_power_law(xs, ys)
    assert abs(a - 2.0) < 1e-6
    assert abs(b - (-0.5)) < 1e-6


def test_predict_interpolates():
    a, b = 2.0, -0.5
    assert abs(predict((a, b), 4) - 1.0) < 1e-9


def test_fit_power_law_rejects_invalid_input():
    with pytest.raises(ValueError):
        fit_power_law([1], [1.0])
    with pytest.raises(ValueError):
        fit_power_law([1, 2], [1.0, -1.0])
