import torch

from llmbepees_cl.basics import make_blobs, train_test_split


def test_make_blobs_shapes_dtypes_labels():
    x, y = make_blobs(n_samples=90, n_centers=3, seed=0)
    assert x.shape == (90, 2)
    assert x.dtype == torch.float32
    assert y.dtype == torch.int64
    assert set(y.tolist()) == {0, 1, 2}


def test_make_blobs_deterministic():
    x1, y1 = make_blobs(n_samples=50, seed=7)
    x2, y2 = make_blobs(n_samples=50, seed=7)
    assert torch.equal(x1, x2)
    assert torch.equal(y1, y2)


def test_train_test_split_sizes():
    x, y = make_blobs(n_samples=100, seed=0)
    x_train, y_train, x_test, y_test = train_test_split(x, y, test_frac=0.2, seed=0)
    assert x_train.shape[0] == 80
    assert x_test.shape[0] == 20
    assert y_train.shape[0] == 80
    assert y_test.shape[0] == 20
