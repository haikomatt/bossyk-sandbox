"""Binomial confidence interval on a domain's violation base rate
(Amendment 2b(i): "every results table reports each domain's violation base
rate with a binomial CI").

Wilson score interval, not the normal (Wald) approximation: the normal
approximation is unreliable near p=0/1 and at the item counts here (a few
hundred positives out of a few thousand), and Wilson is the standard,
closed-form fix. No scipy dependency (not installed in this repo's venv --
`uv run python -c "import scipy"` fails); stdlib `math` only, matching the
rest of this package's "pure numpy/stdlib" convention.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# 97.5th percentile of the standard normal, for a 95% two-sided interval.
Z_95 = 1.959963984540054


@dataclass(frozen=True)
class BinomialCI:
    n: int
    n_pos: int
    rate: float
    lower: float
    upper: float
    z: float


def wilson_interval(n_pos: int, n: int, *, z: float = Z_95) -> BinomialCI:
    """Wilson score interval for a binomial proportion `n_pos / n` at
    confidence level implied by `z` (default z=1.96 -> 95%).

    `n == 0` returns a degenerate `(nan, nan, nan)` CI -- an honest "not
    defined", not a spurious 0."""
    if n_pos < 0 or n < 0 or n_pos > n:
        raise ValueError(f"invalid n_pos={n_pos}, n={n}")
    if n == 0:
        return BinomialCI(
            n=0, n_pos=0, rate=float("nan"), lower=float("nan"), upper=float("nan"), z=z
        )
    p = n_pos / n
    denom = 1.0 + z * z / n
    centre = p + z * z / (2 * n)
    half_width = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    lower = (centre - half_width) / denom
    upper = (centre + half_width) / denom
    return BinomialCI(
        n=n,
        n_pos=n_pos,
        rate=p,
        lower=max(0.0, lower),
        upper=min(1.0, upper),
        z=z,
    )
