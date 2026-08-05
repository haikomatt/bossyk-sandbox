from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from bossyk_sandbox.instruments.base import Instrument
from bossyk_sandbox.instruments.policy import default_policy_path
from bossyk_sandbox.scenarios.loader import SCENARIOS_PATH
from bossyk_sandbox.scenarios.runner import (
    advice_eligibility_fast_rules,
    advice_fast_rules,
    default_fast_rules,
    outreach_fast_rules,
    retail_fast_rules,
)


@dataclass(frozen=True)
class DomainConfig:
    """Bundles the per-domain wiring `scripts/benchmark_run.py` needs:
    which scenario set to load, which policy to judge against, and which
    fast-path rules gate the domain's tools. Each field mirrors an existing
    parameterised function/constant (`scenarios.loader.load_scenarios`,
    `instruments.policy.build_default_policy_instrument`,
    `scenarios.runner.run_scenario`'s `fast_rules`) — this module only
    bundles them per domain, it does not change their behaviour."""

    name: str
    scenarios_path: Path
    policy_path: Path
    fast_rules_factory: Callable[[], list[Instrument]]


def _airline_domain() -> DomainConfig:
    return DomainConfig(
        name="airline",
        scenarios_path=SCENARIOS_PATH,
        policy_path=default_policy_path(),
        fast_rules_factory=default_fast_rules,
    )


def _retail_domain() -> DomainConfig:
    return DomainConfig(
        name="retail",
        scenarios_path=SCENARIOS_PATH.parent.parent / "retail" / "scenarios.json",
        policy_path=default_policy_path().parent / "retail-support-v1.yaml",
        fast_rules_factory=retail_fast_rules,
    )


def _outreach_domain() -> DomainConfig:
    return DomainConfig(
        name="outreach",
        scenarios_path=SCENARIOS_PATH.parent.parent / "outreach" / "scenarios.json",
        policy_path=default_policy_path().parent / "outreach-outbound-v1.yaml",
        fast_rules_factory=outreach_fast_rules,
    )


def _advice_domain() -> DomainConfig:
    return DomainConfig(
        name="advice",
        scenarios_path=SCENARIOS_PATH.parent.parent / "advice" / "scenarios.json",
        policy_path=default_policy_path().parent / "advice-support-v1.yaml",
        fast_rules_factory=advice_fast_rules,
    )


def _advice_eligibility_domain() -> DomainConfig:
    """The detector-training transfer domain (advice-eligibility-domain-spec.md):
    extends the advice scaffolding above with a fresh scenario set + fast
    rules, but REUSES `advice-support-v1.yaml` (extend, don't fork -- the
    policy adds one prohibited-action line for the eligibility-gating
    boundary, judged by the same prose judge as the "advice" domain)."""
    return DomainConfig(
        name="advice-eligibility",
        scenarios_path=SCENARIOS_PATH.parent.parent / "advice" / "eligibility-scenarios.json",
        policy_path=default_policy_path().parent / "advice-support-v1.yaml",
        fast_rules_factory=advice_eligibility_fast_rules,
    )


# Builders, not built instances -- each is called fresh inside `domain_config`
# so a `BOSSYK_ROOT` override (or monkeypatch in a test) is honored at
# lookup time rather than baked in at import time.
_DOMAIN_BUILDERS: dict[str, Callable[[], DomainConfig]] = {
    "airline": _airline_domain,
    "retail": _retail_domain,
    "outreach": _outreach_domain,
    "advice": _advice_domain,
    "advice-eligibility": _advice_eligibility_domain,
}


def domain_config(name: str) -> DomainConfig:
    """Looks up a registered domain's config. Raises `KeyError` for an
    unregistered domain name — callers should let it propagate rather than
    silently falling back to a default domain."""
    return _DOMAIN_BUILDERS[name]()
