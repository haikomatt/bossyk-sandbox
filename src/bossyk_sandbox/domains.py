from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from bossyk_sandbox.instruments.base import Instrument
from bossyk_sandbox.instruments.policy import DEFAULT_POLICY_PATH
from bossyk_sandbox.scenarios.loader import SCENARIOS_PATH
from bossyk_sandbox.scenarios.runner import default_fast_rules, retail_fast_rules


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


AIRLINE = DomainConfig(
    name="airline",
    scenarios_path=SCENARIOS_PATH,
    policy_path=DEFAULT_POLICY_PATH,
    fast_rules_factory=default_fast_rules,
)

RETAIL = DomainConfig(
    name="retail",
    scenarios_path=SCENARIOS_PATH.parent.parent / "retail" / "scenarios.json",
    policy_path=DEFAULT_POLICY_PATH.parent / "retail-support-v1.yaml",
    fast_rules_factory=retail_fast_rules,
)

DOMAINS: dict[str, DomainConfig] = {"airline": AIRLINE, "retail": RETAIL}


def domain_config(name: str) -> DomainConfig:
    """Looks up a registered domain's config. Raises `KeyError` for an
    unregistered domain name — callers should let it propagate rather than
    silently falling back to a default domain."""
    return DOMAINS[name]
