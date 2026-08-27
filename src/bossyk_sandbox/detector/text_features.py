"""Deterministic bag-of-words text features for the frozen detector baselines.

Two vectorisers, per the pre-registered dossier's confound 2 (BoW OOV
fairness -- `Hypotheses/a-fine-tuned-violation-detector-transfers-cross-domain.md`):

- `hashing_vectorize`: unbounded vocabulary via md5-bucket hashing, so a
  transfer miss can never be an out-of-vocabulary artefact. Extracted here
  from `scripts/text_baseline.py` (same implementation, same convention:
  Python's built-in `hash()` is per-process salted, so md5 is used instead
  for byte-identical buckets across processes/runs) so the H2 probe control
  and the frozen baselines share one implementation rather than two
  independently-maintained copies.
- `fit_fixed_vocab` / `fixed_vocab_vectorize`: a real, bounded vocabulary
  fit on TRAIN ONLY; any token unseen at fit time is OOV (silently dropped)
  at transform time on other splits/domains. This is the "handicapped"
  vectoriser the confound gate exists to bound -- reported ALONGSIDE the
  hashing vectoriser, never instead of it.

Pure numpy + stdlib; no sklearn, no network.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]

_TOKEN_RE = re.compile(r"[a-z0-9#]+")


def tokenize(text: str) -> list[str]:
    """Lower-case word/number tokens -- the same simple tokenisation used
    throughout this repo's text baselines (`scripts/text_baseline.py`)."""
    return _TOKEN_RE.findall(text.lower())


def hashing_vectorize(texts: list[str], *, n_features: int = 4096) -> Array:
    """Deterministic bag-of-words hashing vectorizer (md5 -> bucket): stable
    across processes, unbounded vocabulary, so a transfer miss is never an
    out-of-vocabulary artefact of a fitted vocabulary."""
    x = np.zeros((len(texts), n_features), dtype=np.float64)
    for i, text in enumerate(texts):
        for tok in tokenize(text):
            bucket = int(hashlib.md5(tok.encode()).hexdigest(), 16) % n_features
            x[i, bucket] += 1.0
    return x


@dataclass(frozen=True)
class FixedVocab:
    """A vocabulary fit on TRAIN texts only. `tokens[i]` is the token at
    feature index `i`; any token not present here is OOV at transform time."""

    tokens: tuple[str, ...]

    @property
    def size(self) -> int:
        return len(self.tokens)


def fit_fixed_vocab(texts: list[str], *, max_features: int = 4096, min_df: int = 2) -> FixedVocab:
    """Fit a fixed vocabulary from TRAIN texts only: keep tokens with document
    frequency >= `min_df`, ranked by frequency descending (ties broken
    alphabetically for determinism), truncated to `max_features`. Never see
    val/test/OOD text -- that is precisely the OOV-fairness confound the
    hashing vectoriser exists to bound."""
    df: Counter[str] = Counter()
    for text in texts:
        df.update(set(tokenize(text)))
    kept = [tok for tok, count in df.items() if count >= min_df]
    kept.sort(key=lambda t: (-df[t], t))
    return FixedVocab(tokens=tuple(kept[:max_features]))


def fixed_vocab_vectorize(vocab: FixedVocab, texts: list[str]) -> Array:
    """Count-vectorize `texts` against a previously-fit `FixedVocab`. Tokens
    absent from the vocabulary are silently dropped (OOV) -- the fairness
    confound this vectoriser is deliberately reported alongside the hashing
    one to bound."""
    index = {tok: i for i, tok in enumerate(vocab.tokens)}
    x = np.zeros((len(texts), vocab.size), dtype=np.float64)
    for i, text in enumerate(texts):
        for tok in tokenize(text):
            j = index.get(tok)
            if j is not None:
                x[i, j] += 1.0
    return x
