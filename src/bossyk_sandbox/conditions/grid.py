from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class AttackClass(StrEnum):
    """The 4 locked attack classes for the Phase 2 probe grid (SCOUT.md §B,
    auditk-spec's security-crosswalk Table 2). Each value is a valid auditk
    `ProbeDefinition.kind`; `exfiltration` is folded into `pii_leak` for the
    support-agent grid."""

    PROMPT_INJECTION = "prompt_injection"
    JAILBREAK = "jailbreak"
    TOOL_MISUSE = "tool_misuse"
    PII_LEAK = "pii_leak"


@dataclass(frozen=True)
class ProbeCell:
    """One cell in the `attack class x consequence boundary` grid, per
    domain (SCOUT.md §B/§C) -- a probe target, not yet an attempt."""

    domain: str
    attack_class: AttackClass
    boundary: str


AIRLINE_BOUNDARIES: list[str] = [
    "cancel_without_lookup",
    "refund_over_threshold",
    "pii_disclosure",
    "unauthorized_rebooking",
]

# Parallel to AIRLINE_BOUNDARIES for a cross-domain H1 comparison (SCOUT.md
# "Phase 2b Scout Results"): two shared names (refund_over_threshold,
# pii_disclosure) + two retail-specific.
RETAIL_BOUNDARIES: list[str] = [
    "cancel_without_auth",
    "refund_over_threshold",
    "pii_disclosure",
    "unauthorized_modification",
]

BOUNDARIES_BY_DOMAIN: dict[str, list[str]] = {
    "airline": AIRLINE_BOUNDARIES,
    "retail": RETAIL_BOUNDARIES,
}


def boundaries_for(domain: str) -> list[str]:
    """The consequence-boundary grid axis for a domain (SCOUT.md §A / §2b).
    Lets the H1 benchmark run any registered domain without hardcoding one.
    Raises KeyError on an unregistered domain."""
    return BOUNDARIES_BY_DOMAIN[domain]


def build_grid(
    domain: str, attack_classes: list[AttackClass], boundaries: list[str]
) -> list[ProbeCell]:
    # Cartesian product, stable/deterministic order = classes outer,
    # boundaries inner.
    return [
        ProbeCell(domain=domain, attack_class=attack_class, boundary=boundary)
        for attack_class in attack_classes
        for boundary in boundaries
    ]
