import pytest
import torch

from llmbepees_cl.minigpt import (
    CharTokenizer,
    MiniGPT,
    encode_corpus,
    estimate_loss,
    generate,
    get_batch,
    train_language_model,
)


def test_get_batch_shifts_targets_by_one():
    data = torch.arange(100)
    x, y = get_batch(data, block_size=8, batch_size=4, generator=torch.Generator().manual_seed(0))
    assert x.shape == (4, 8) and y.shape == (4, 8)
    assert torch.equal(y, x + 1)


def test_get_batch_rejects_short_data():
    with pytest.raises(ValueError):
        get_batch(torch.arange(4), block_size=8, batch_size=1)


def test_training_reduces_loss_on_repetitive_text():
    text = "abcabcabc" * 60
    tok = CharTokenizer.from_text(text)
    data = encode_corpus(text, tok)
    model = MiniGPT(vocab_size=tok.vocab_size, n_embd=32, n_head=2, n_layer=1, block_size=8)

    history = train_language_model(
        model, data, steps=200, batch_size=16, block_size=8, lr=1e-2, seed=0
    )
    first = sum(history[:10]) / 10
    last = sum(history[-10:]) / 10
    assert last < first


def test_generate_extends_sequence_by_requested_tokens():
    tok = CharTokenizer.from_text("abc")
    model = MiniGPT(vocab_size=tok.vocab_size, n_embd=8, n_head=2, n_layer=1, block_size=8)
    prompt = torch.tensor([[tok.stoi["a"]]])
    out = generate(model, prompt, max_new_tokens=5, generator=torch.Generator().manual_seed(0))
    assert out.shape == (1, 6)


def test_estimate_loss_is_finite():
    tok = CharTokenizer.from_text("abcabc")
    data = encode_corpus("abcabc" * 20, tok)
    model = MiniGPT(vocab_size=tok.vocab_size, n_embd=8, n_head=2, n_layer=1, block_size=4)
    loss = estimate_loss(model, data, block_size=4, batch_size=4, batches=3)
    assert loss > 0
