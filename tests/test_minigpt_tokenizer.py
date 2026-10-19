import pytest

from llmbepees_cl.minigpt import BPETokenizer, CharTokenizer

TEXT = "abracadabra abracadabra"


def test_char_roundtrip():
    tok = CharTokenizer.from_text(TEXT)
    ids = tok.encode(TEXT)
    assert tok.decode(ids) == TEXT


def test_char_vocab_is_sorted_unique():
    tok = CharTokenizer.from_text("banana")
    assert tok.itos == ["a", "b", "n"]
    assert tok.vocab_size == 3
    assert len(tok) == 3


def test_char_unknown_character_raises():
    tok = CharTokenizer.from_text("abc")
    with pytest.raises(KeyError):
        tok.encode("z")


def test_bpe_roundtrip():
    tok = BPETokenizer().train(TEXT, n_merges=20)
    ids = tok.encode(TEXT)
    assert tok.decode(ids) == TEXT


def test_bpe_compresses_and_grows_vocab():
    tok = BPETokenizer().train(TEXT, n_merges=20)
    base_vocab = len(set(TEXT))
    assert tok.vocab_size > base_vocab
    assert len(tok.encode(TEXT)) < len(TEXT)


def test_bpe_more_merges_give_fewer_tokens():
    few = BPETokenizer().train(TEXT, n_merges=2)
    many = BPETokenizer().train(TEXT, n_merges=20)
    assert len(many.encode(TEXT)) <= len(few.encode(TEXT))
