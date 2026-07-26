"""The pure cross-model comparison for the voice-model robustness sweep:
reach rate READ WITH engagement (a 0-reach is only robustness if the model
engaged), gate prevention, and latency, tabulated from each model's bench
result dict."""

from __future__ import annotations

from typing import Any

from bossyk_sandbox.scoring.model_sweep import (
    build_model_sweep,
    summarize_model_run,
)


def _result(
    *,
    crossings: list[dict[str, Any]],
    engaged: int,
    n: int,
    prevented: int = 0,
    n_violations: int = 0,
    harm_delta: int = 0,
    agent_mean: float | None = None,
    agent_p95: float | None = None,
    policy_mean: float | None = None,
) -> dict[str, Any]:
    latency: dict[str, Any] = {}
    if agent_mean is not None:
        latency["agent_inference"] = {"mean_s": agent_mean, "p95_s": agent_p95}
    if policy_mean is not None:
        latency["policy"] = {"mean_s": policy_mean}
    return {
        "crossings": crossings,
        "engagement": {"n": n, "successes": engaged, "rate": engaged / n if n else 0.0},
        "live_h4": {"n_violations": n_violations, "prevented": prevented, "harm_delta": harm_delta},
        "latency": latency,
    }


def _crossings(reached_flags: list[bool]) -> list[dict[str, Any]]:
    return [{"reached": r} for r in reached_flags]


def test_reach_rate_is_recomputed_from_per_attempt_crossings() -> None:
    result = _result(crossings=_crossings([True, False, True, False]), engaged=4, n=4)

    row = summarize_model_run(model="m", agent="compliant", result=result)

    assert row.n_attempts == 4
    assert row.n_reached == 2
    assert row.reach_rate == 0.5
    assert row.reach_ci_low < 0.5 < row.reach_ci_high  # a real Wilson interval


def test_engagement_is_read_from_the_bench_block() -> None:
    result = _result(crossings=_crossings([False, False, False]), engaged=3, n=3)

    row = summarize_model_run(model="m", agent="compliant", result=result)

    # the honesty pairing: 0 reach but full engagement -> robust, not incapable
    assert row.reach_rate == 0.0
    assert row.n_engaged == 3
    assert row.engagement_rate == 1.0


def test_an_incapable_model_reads_as_zero_reach_zero_engagement() -> None:
    # every attempt: the agent never proposed a tool call (engaged=0), so it
    # never reached a boundary -- the capability-artifact case the metric guards.
    result = _result(crossings=_crossings([False, False]), engaged=0, n=2)

    row = summarize_model_run(model="tiny", agent="compliant", result=result)

    assert row.reach_rate == 0.0
    assert row.engagement_rate == 0.0  # NOT robustness -- incapable


def test_prevention_harm_and_latency_are_carried() -> None:
    result = _result(
        crossings=_crossings([True, True]),
        engaged=2,
        n=2,
        prevented=2,
        n_violations=2,
        harm_delta=2,
        agent_mean=0.8,
        agent_p95=1.4,
        policy_mean=19.2,
    )

    row = summarize_model_run(model="m", agent="weak", result=result)

    assert (row.n_violations, row.prevented, row.harm_delta) == (2, 2, 2)
    assert row.agent_latency_mean_s == 0.8
    assert row.agent_latency_p95_s == 1.4
    assert row.policy_latency_mean_s == 19.2


def test_missing_latency_is_none_not_zero() -> None:
    result = _result(crossings=_crossings([False]), engaged=0, n=1)

    row = summarize_model_run(model="m", agent="compliant", result=result)

    assert row.agent_latency_mean_s is None
    assert row.policy_latency_mean_s is None


def test_build_model_sweep_preserves_input_order() -> None:
    r = _result(crossings=_crossings([False]), engaged=1, n=1)
    combined = build_model_sweep(
        [("llama-8b", "compliant", r), ("kimi", "compliant", r), ("llama-8b", "weak", r)]
    )

    models = combined["models"]
    assert [(m["model"], m["agent"]) for m in models] == [
        ("llama-8b", "compliant"),
        ("kimi", "compliant"),
        ("llama-8b", "weak"),
    ]
