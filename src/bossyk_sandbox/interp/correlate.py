"""Does the behavioral uncertainty signal separate policy violations? (pure).

The correlation layer -- the point of the behavioral phase. Given per-step pairs
of (StepUncertainty, is_violation), where `is_violation` comes from the Gate's
policy verdict, it asks per uncertainty feature: *do violation steps score
differently from compliant steps?*

Measured with rank **AUROC** (equivalently the Mann-Whitney U statistic):
`P(a random violation step's feature > a random compliant step's feature)`, with
ties counted as half. Chosen over a t-test / Cohen's d because it is
threshold-free, makes no normality assumption, and is stable on the small,
skewed samples a live tau2 run produces.

Reading the number:
- **0.5** -- no separation; the feature carries no violation signal.
- **> 0.5** -- higher feature value associates with violation (expected for
  surprisal/entropy: the model is *less* certain when it violates).
- **< 0.5** -- lower feature value associates with violation, equally
  informative (expected for `min_margin`: a violation step may be a *near-tie*,
  i.e. a smaller margin). Interpret strength as the distance |AUROC - 0.5|.

This is a descriptive separation measure, not the Phase-4 probe and not the H2
text-baseline comparison -- it answers "is there any behavioral signal at all"
before the (expensive, activation-based) probe is built.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from bossyk_sandbox.interp.logprob_metrics import StepUncertainty

# The uncertainty features scored against the policy label, with how to read a
# non-0.5 AUROC. Order is the report order.
_FEATURES: tuple[tuple[str, Callable[[StepUncertainty], float]], ...] = (
    ("max_surprisal", lambda s: s.max_surprisal),
    ("mean_surprisal", lambda s: s.mean_surprisal),
    ("mean_entropy", lambda s: s.mean_entropy),
    ("min_margin", lambda s: s.min_margin),
)


@dataclass(frozen=True)
class FeatureSeparation:
    """How well one uncertainty feature separates violation from compliant steps."""

    feature: str
    n_violation: int
    n_compliant: int
    mean_violation: float
    mean_compliant: float
    auroc: float  # nan when either class is empty (separation undefined)


def auroc(scores: Sequence[float], labels: Sequence[bool]) -> float:
    """Rank AUROC = P(positive score > negative score) + 0.5 * P(tie).

    `labels[i]` True marks the positive (violation) class. Computed from average
    ranks (exact tie handling) via the Mann-Whitney identity, so no threshold
    sweep is needed. Returns `nan` when either class is empty -- separation is
    undefined, and the caller/report must say so rather than print a spurious
    0.5. Raises `ValueError` on length mismatch."""
    if len(scores) != len(labels):
        raise ValueError("scores and labels must be the same length")
    n = len(scores)
    n_pos = sum(1 for label in labels if label)
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return math.nan

    order = sorted(range(n), key=lambda i: scores[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        average_rank = (i + j) / 2.0 + 1.0  # 1-based rank, averaged over the tie block
        for k in range(i, j + 1):
            ranks[order[k]] = average_rank
        i = j + 1

    sum_ranks_pos = math.fsum(ranks[i] for i in range(n) if labels[i])
    u = sum_ranks_pos - n_pos * (n_pos + 1) / 2.0
    return u / (n_pos * n_neg)


def _mean(values: list[float]) -> float:
    return math.fsum(values) / len(values) if values else math.nan


def correlate_uncertainty(
    pairs: Sequence[tuple[StepUncertainty, bool]],
) -> list[FeatureSeparation]:
    """Per-feature separation of violation vs compliant steps over aligned pairs.

    Each pair is (per-step uncertainty summary, is_violation). Steps with
    `n_tokens == 0` (pure tool-call turns that produced no scored tokens) carry
    no behavioral signal and are **excluded** -- their zero-filled metrics would
    otherwise dilute the separation. Producing the (uncertainty, label) alignment
    from a live session is the caller's job (Phase 2 e2e); this function is only
    the statistics."""
    scored = [(u, v) for (u, v) in pairs if u.n_tokens > 0]
    labels = [v for (_, v) in scored]
    results: list[FeatureSeparation] = []
    for name, getter in _FEATURES:
        values = [getter(u) for (u, _) in scored]
        viol = [x for x, v in zip(values, labels, strict=True) if v]
        ok = [x for x, v in zip(values, labels, strict=True) if not v]
        results.append(
            FeatureSeparation(
                feature=name,
                n_violation=len(viol),
                n_compliant=len(ok),
                mean_violation=_mean(viol),
                mean_compliant=_mean(ok),
                auroc=auroc(values, labels),
            )
        )
    return results


def format_separation_table(rows: Sequence[FeatureSeparation]) -> str:
    """Render the separation results as a small Markdown table for a report."""
    header = (
        "| feature | n_viol | n_ok | mean(viol) | mean(ok) | AUROC |\n|---|---|---|---|---|---|"
    )
    lines = [header]
    for r in rows:
        lines.append(
            f"| {r.feature} | {r.n_violation} | {r.n_compliant} | "
            f"{r.mean_violation:.3f} | {r.mean_compliant:.3f} | {r.auroc:.3f} |"
        )
    return "\n".join(lines)
