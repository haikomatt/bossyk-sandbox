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
