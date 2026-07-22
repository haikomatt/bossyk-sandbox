from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol


class GuardrailStrength(StrEnum):
    """How aggressively the model-layer guardrail flags content, before the
    GATE and the slow instruments ever see it (SCOUT.md §D)."""

    OFF = "off"
    LEAKY = "leaky"
    MODERATE = "moderate"
    STRICT = "strict"


@dataclass(frozen=True)
class GuardrailVerdict:
    flagged: bool
    reason: str


class Guardrail(Protocol):
    """Input filter upstream of the GATE, scoring injected condition/model-
    call content -- not a tool-call instrument."""

    def inspect(self, content: str) -> GuardrailVerdict: ...


@dataclass
class GradedRuleGuardrail:
    """Deterministic graded rule/keyword classifier -- the backbone
    `Guardrail` implementation. Strength selects the ruleset/threshold
    applied; higher strengths must flag everything a lower strength flags
    (STRICT is a superset of LEAKY)."""

    strength: GuardrailStrength = GuardrailStrength.LEAKY

    def inspect(self, content: str) -> GuardrailVerdict:
        raise NotImplementedError


def build_default_guardrail(strength: GuardrailStrength = GuardrailStrength.LEAKY) -> Guardrail:
    # A model-backed classifier (HF jailbreak/injection model) can later sit
    # behind this same `Guardrail` interface for the real run -- not built here.
    raise NotImplementedError
