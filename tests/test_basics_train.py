from llmbepees_cl.basics import (
    build_mlp,
    evaluate_accuracy,
    make_blobs,
    train_classifier,
    train_test_split,
)


def test_training_reduces_loss_and_learns():
    x, y = make_blobs(n_samples=300, n_centers=3, cluster_std=0.5, seed=0)
    x_train, y_train, x_test, y_test = train_test_split(x, y, seed=0)

    model = build_mlp(in_features=2, n_classes=3, hidden=(16,), seed=0)
    history = train_classifier(model, x_train, y_train, epochs=30, lr=0.1, seed=0)

    assert history[-1] < history[0]
    assert evaluate_accuracy(model, x_test, y_test) > 0.8
