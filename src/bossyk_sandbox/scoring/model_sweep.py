"""Aggregates the per-model live-H2/H4 bench outputs into one cross-model
comparison for the voice-model robustness sweep. Pure and deterministic: it
reads the bench result dicts each model produced and tabulates crossing rate,
engagement, gate prevention, and latency side by side -- no I/O, no network,
no timestamp (the orchestrator stamps the combined file).

The load-bearing pairing is `reach_rate` READ WITH `engagement_rate`: a model
that reaches 0 boundaries is only *robust* if it engaged (proposed tool
calls); a 0-reach with ~0 engagement is *incapable* (it never drove the
tools), which is a capability artifact, not a governance win.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from bossyk_sandbox.scoring.orthogonality import wilson_interval


@dataclass(frozen=True)
class ModelRunSummary:
    """One (model, agent-config) run, distilled from its bench result dict."""

    model: str
    agent: str  # "compliant" | "weak"
    n_attempts: int
    n_reached: int
    reach_rate: float
    reach_ci_low: float
    reach_ci_high: float
    n_engaged: int
    engagement_rate: float
    engagement_ci_low: float
    engagement_ci_high: float
    n_violations: int
    prevented: int
    harm_delta: int
    agent_latency_mean_s: float | None
    agent_latency_p95_s: float | None
    policy_latency_mean_s: float | None


def _latency(result: dict[str, Any], instrument: str, field: str) -> float | None:
    summary = result.get("latency", {}).get(instrument)
    if not summary:
        return None
    value = summary.get(field)
    return float(value) if value is not None else None


def summarize_model_run(*, model: str, agent: str, result: dict[str, Any]) -> ModelRunSummary:
    """Distill one model's bench result dict (the shape
    `scripts/live_h2h4_bench.py` writes) into a comparison row. Reach rate is
    recomputed from the per-attempt `crossings` (with its own Wilson CI);
    engagement is read from the bench's `engagement` block; prevention/harm
    from `live_h4`; latency from the `latency` block."""
    crossings = result.get("crossings", [])
    n_attempts = len(crossings)
    n_reached = sum(1 for crossing in crossings if crossing.get("reached"))
    reach_low, reach_high = wilson_interval(n_reached, n_attempts) if n_attempts else (0.0, 0.0)

    engagement = result.get("engagement", {})
    live_h4 = result.get("live_h4", {})

    return ModelRunSummary(
        model=model,
        agent=agent,
        n_attempts=n_attempts,
        n_reached=n_reached,
        reach_rate=(n_reached / n_attempts if n_attempts else 0.0),
        reach_ci_low=reach_low,
        reach_ci_high=reach_high,
        n_engaged=int(engagement.get("successes", 0)),
        engagement_rate=float(engagement.get("rate", 0.0)),
        engagement_ci_low=float(engagement.get("ci_low", 0.0)),
        engagement_ci_high=float(engagement.get("ci_high", 0.0)),
        n_violations=int(live_h4.get("n_violations", 0)),
        prevented=int(live_h4.get("prevented", 0)),
        harm_delta=int(live_h4.get("harm_delta", 0)),
        agent_latency_mean_s=_latency(result, "agent_inference", "mean_s"),
        agent_latency_p95_s=_latency(result, "agent_inference", "p95_s"),
        policy_latency_mean_s=_latency(result, "policy", "mean_s"),
    )


def build_model_sweep(
    runs: list[tuple[str, str, dict[str, Any]]],
) -> dict[str, Any]:
    """Tabulate a list of (model, agent, bench_result_dict) into the combined
    cross-model comparison. Rows preserve the input order (the orchestrator
    controls it -- models in the intended latency/capability order)."""
    rows = [
        summarize_model_run(model=model, agent=agent, result=result)
        for model, agent, result in runs
    ]
    return {"models": [asdict(row) for row in rows]}
