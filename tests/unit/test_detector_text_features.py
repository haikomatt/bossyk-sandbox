"""Unit tests for bossyk_sandbox.detector.text_features."""

from __future__ import annotations

from bossyk_sandbox.detector.text_features import (
    fit_fixed_vocab,
    fixed_vocab_vectorize,
    hashing_vectorize,
    tokenize,
)


def test_tokenize_lowercases_and_splits_on_non_word_chars() -> None:
    assert tokenize("Cancel Order #123, NOW!") == ["cancel", "order", "#123", "now"]


def test_hashing_vectorize_is_deterministic_across_calls() -> None:
    texts = ["cancel my order please", "refund the flight booking"]
    a = hashing_vectorize(texts, n_features=64)
    b = hashing_vectorize(texts, n_features=64)
    assert (a == b).all()


def test_hashing_vectorize_shape_and_nonzero_rows() -> None:
    texts = ["hello world", "another example sentence"]
    x = hashing_vectorize(texts, n_features=128)
    assert x.shape == (2, 128)
    assert (x.sum(axis=1) > 0).all()


def test_hashing_vectorize_same_token_same_bucket_regardless_of_text() -> None:
    # "order" must hash to the same bucket wherever it appears.
    x = hashing_vectorize(["order status", "please check order"], n_features=256)
    import hashlib

    bucket = int(hashlib.md5(b"order").hexdigest(), 16) % 256
    assert x[0, bucket] == 1.0
    assert x[1, bucket] == 1.0


def test_fixed_vocab_only_fit_on_train_texts() -> None:
    train_texts = ["cancel order", "cancel order", "refund item"]
    vocab = fit_fixed_vocab(train_texts, max_features=10, min_df=1)
    assert "cancel" in vocab.tokens
    assert "order" in vocab.tokens
    assert "refund" in vocab.tokens


def test_fixed_vocab_min_df_drops_rare_tokens() -> None:
    train_texts = ["cancel order", "cancel order", "onceword here"]
    vocab = fit_fixed_vocab(train_texts, max_features=10, min_df=2)
    assert "cancel" in vocab.tokens
    assert "onceword" not in vocab.tokens  # appears in only 1 doc, min_df=2


def test_fixed_vocab_max_features_truncates_by_frequency_then_alpha() -> None:
    train_texts = ["zzz zzz aaa aaa mmm mmm", "zzz aaa mmm"]
    vocab = fit_fixed_vocab(train_texts, max_features=2, min_df=1)
    # all three tokens tie on doc-frequency (2 docs each); alpha tiebreak keeps
    # the first two alphabetically.
    assert vocab.tokens == ("aaa", "mmm")


def test_fixed_vocab_transform_drops_oov_tokens_silently() -> None:
    vocab = fit_fixed_vocab(["cancel order"], max_features=10, min_df=1)
    x = fixed_vocab_vectorize(vocab, ["cancel order refund"])  # "refund" is OOV
    assert x.shape == (1, vocab.size)
    assert x.sum() == 2.0  # only "cancel" and "order" counted


def test_fixed_vocab_vectorize_shape_matches_vocab_size() -> None:
    vocab = fit_fixed_vocab(["a b c", "a b"], max_features=10, min_df=1)
    x = fixed_vocab_vectorize(vocab, ["a b c", "z z z"])
    assert x.shape == (2, vocab.size)
    assert x[1].sum() == 0.0  # all-OOV row is all-zero, not an error
