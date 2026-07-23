from __future__ import annotations

from auditk.schema import ProbeDefinition

from bossyk_sandbox.conditions.adversary import Adversary, ProbeAttempt
from bossyk_sandbox.conditions.grid import ProbeCell
from bossyk_sandbox.conditions.retention import freeze_attempt


def generate_grounded_attempts(
    adversary: Adversary, cells: list[ProbeCell], budget: int
) -> list[ProbeAttempt]:
    """Run `adversary` over every cell (once per cell, `budget` attempts
    each) and return ALL attempts -- ok, refused, and error alike -- so the
    caller can build a token ledger over the full set (refusals cost tokens
    too) before freezing only the usable ones. See `generate_grounded_corpus`."""
    attempts: list[ProbeAttempt] = []
    for cell in cells:
        attempts.extend(adversary.generate_attempts(cell, budget))
    return attempts


def generate_grounded_corpus(
    adversary: Adversary, cells: list[ProbeCell], budget: int
) -> list[ProbeDefinition]:
    """Generate a live-replayable grounded-attack corpus: run `adversary`
    over the grid and freeze every OK attempt into a `ProbeDefinition`
    (`retention.freeze_attempt`). Refused/error attempts carry no payload and
    are dropped -- never a live-corpus probe. The adversary should be grounded
    (conditions.fireworks_adversary tool_context) so the frozen payloads
    reference the agent's real tools."""
    return [
        freeze_attempt(attempt)
        for attempt in generate_grounded_attempts(adversary, cells, budget)
        if attempt.status == "ok"
    ]
