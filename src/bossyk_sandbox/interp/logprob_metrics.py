"""Per-token logprob -> per-step uncertainty summary (pure, stdlib only).

The behavioral interpretability signal. The agent's OpenAI-compatible endpoint
(vLLM on the pod) can return, per generated token, the chosen token's logprob
plus the top-k alternatives' logprobs. This module turns that raw structure into
three token-level quantities and a per-step summary the correlation layer scores
against the Gate's policy verdict:

- **surprisal** -- how unexpected the chosen token was (nats). High surprisal =
  the model committed to something its own distribution disfavoured.
- **top-k entropy** -- how spread the (visible, top-k) distribution was at that
  position. Renormalised over the visible top-k, so it is honestly a *truncated*
  entropy, not the full-vocab entropy.
- **margin** -- gap between the best and second-best alternative (nats). A small
  margin is a near-tie, i.e. low local confidence.

No torch/numpy: OpenAI logprobs are small top-k lists, so plain `math` + `fsum`
is exact enough and keeps this module installable with zero heavy deps (it runs
in the default hermetic test suite). Logprobs are natural log, matching the
OpenAI/vLLM convention.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TokenLogprob:
    """One generated token's logprob evidence.

    `logprob` is the chosen token's log-probability. `top_logprobs` are the
    log-probabilities of the top-k alternatives at that position (the OpenAI
    `top_logprobs` field), highest first is NOT assumed -- callers may pass them
    in any order; the metrics sort where order matters. `top_logprobs` may be
    empty when the endpoint was not asked for alternatives, in which case
    entropy and margin are undefined and reported as 0.0.
    """

    logprob: float
    top_logprobs: tuple[float, ...] = ()


@dataclass(frozen=True)
class StepUncertainty:
    """Per-step aggregate of the token-level metrics over one agent turn.

    This is the record attached alongside the auditk `Step` and lined up against
    the Gate's policy verdict in the correlation layer. `n_tokens == 0` means the
    turn produced no scored tokens (e.g. a pure tool-call turn with empty
    content); all metrics are 0.0 and the step contributes nothing to the
    correlation.
    """

    n_tokens: int
    mean_surprisal: float
    max_surprisal: float
    mean_entropy: float
    min_margin: float


def surprisal(token: TokenLogprob) -> float:
    """Surprisal (nats) of the chosen token: `-logprob`. Always >= 0 for a valid
    log-probability (logprob <= 0)."""
    return -token.logprob


def top_k_entropy(token: TokenLogprob) -> float:
    """Shannon entropy (nats) of the visible top-k distribution, renormalised.

    The top-k logprobs rarely sum to 1 (the tail is truncated), so we renormalise
    over the visible alternatives and report the entropy of that conditional
    distribution -- a well-defined quantity, honestly labelled *truncated*. `0.0`
    when there are no alternatives (entropy undefined). A degenerate all-mass-on-
    one distribution returns `0.0`.
    """
    if not token.top_logprobs:
        return 0.0
    probs = [math.exp(lp) for lp in token.top_logprobs]
    total = math.fsum(probs)
    if total <= 0.0:
        return 0.0
    return -math.fsum((p / total) * math.log(p / total) for p in probs if p > 0.0)


def margin(token: TokenLogprob) -> float:
    """Confidence margin (nats): `logprob(top1) - logprob(top2)` over the visible
    alternatives. `0.0` when fewer than two alternatives are visible. Larger =
    more locally confident; near-zero = a near-tie."""
    if len(token.top_logprobs) < 2:
        return 0.0
    ranked = sorted(token.top_logprobs, reverse=True)
    return ranked[0] - ranked[1]


def summarize(tokens: Sequence[TokenLogprob]) -> StepUncertainty:
    """Aggregate the token-level metrics over one agent turn.

    Chooses aggregations that surface a *localised* spike: `max_surprisal` and
    `min_margin` catch a single high-uncertainty token that means-based summaries
    would wash out -- the hypothesis being that a policy violation is committed at
    a token, not smeared across the whole turn. An empty turn yields the all-zero,
    `n_tokens == 0` summary.
    """
    if not tokens:
        return StepUncertainty(0, 0.0, 0.0, 0.0, 0.0)
    surprisals = [surprisal(t) for t in tokens]
    entropies = [top_k_entropy(t) for t in tokens]
    margins = [margin(t) for t in tokens]
    n = len(tokens)
    return StepUncertainty(
        n_tokens=n,
        mean_surprisal=math.fsum(surprisals) / n,
        max_surprisal=max(surprisals),
        mean_entropy=math.fsum(entropies) / n,
        min_margin=min(margins),
    )


def parse_openai_logprobs(content: Sequence[dict[str, Any]] | None) -> list[TokenLogprob]:
    """Parse an OpenAI/vLLM ``logprobs.content`` list into ``TokenLogprob``s.

    ``content`` is the per-token list the OpenAI chat API returns under
    ``choices[].logprobs.content``; LangChain surfaces it verbatim at
    ``AIMessage.response_metadata["logprobs"]["content"]``. Each item carries the
    chosen token's ``logprob`` plus a ``top_logprobs`` list of alternatives.
    Missing/None ``top_logprobs`` -> empty (entropy/margin then 0.0). A None or
    empty ``content`` (e.g. a pure tool-call turn that emitted no scored tokens)
    yields an empty list, which ``summarize`` maps to the ``n_tokens == 0``
    summary -- keeping the tool-call-logprobs open question a graceful no-op
    rather than an error.
    """
    tokens: list[TokenLogprob] = []
    for item in content or []:
        alts = item.get("top_logprobs") or []
        tokens.append(
            TokenLogprob(
                logprob=float(item["logprob"]),
                top_logprobs=tuple(float(a["logprob"]) for a in alts),
            )
        )
    return tokens
