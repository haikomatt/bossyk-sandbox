from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from bossyk_sandbox.instruments.base import InstrumentVerdict, ProposedAction

DEFAULT_POLICY_PATH = Path("~/Projects/bossyk/data/policies/airline-support-v1.yaml").expanduser()

ERROR_LABEL = "error"

# bossyk has no pyproject.toml / package metadata (SCOUT.md Phase 1 #2) — it's
# a flat `src/` script collection consumed via sys.path, mirroring bossyk's
# own internal convention (see bossyk/src/run_policy.py). Not an auditk-style
# editable install; this is the reuse path, not a modification.
BOSSYK_SRC = Path("~/Projects/bossyk/src").expanduser()


class PolicyStepResult(Protocol):
    label: str
    reasoning: str


class PolicyJudgeClient(Protocol):
    """Matches bossyk's `PolicyAwareJudge.score_step` — satisfied by the
    real class for the benchmark run, or a fake for deterministic tests."""

    def score_step(
        self, step_id: str, action_text: str, declared_intent: str | None = None
    ) -> PolicyStepResult: ...


def _action_text(proposed: ProposedAction) -> str:
    return f"{proposed.tool_name}({proposed.arguments})"


@dataclass
class PolicyInstrument:
    """Wraps bossyk's `PolicyAwareJudge` as a `SlowInstrument`."""

    judge: PolicyJudgeClient
    name: str = "policy"
    _step_counter: int = field(default=0, repr=False)

    def annotate(
        self, proposed: ProposedAction, history: list[ProposedAction]
    ) -> InstrumentVerdict:
        self._step_counter += 1
        step_id = f"policy-step-{self._step_counter}"
        try:
            result = self.judge.score_step(
                step_id=step_id,
                action_text=_action_text(proposed),
                declared_intent=proposed.declared_intent,
            )
        except Exception as exc:
            # bossyk's PolicyAwareJudge on deepseek-v4-pro has a documented
            # ~28-44% JSON-parse error rate under Fireworks load (bossyk
            # SCOUT.md). A single flaky judge call must not take down the
            # whole benchmark run -- surface it as an explicit "error"
            # verdict instead of propagating.
            return InstrumentVerdict(instrument=self.name, label=ERROR_LABEL, detail=str(exc))
        return InstrumentVerdict(instrument=self.name, label=result.label, detail=result.reasoning)


def build_default_policy_instrument(
    policy_path: Path | str = DEFAULT_POLICY_PATH,
) -> PolicyInstrument:
    """Real judge path: bossyk's `PolicyAwareJudge` against the
    `airline-support-v1` policy, unmodified — including its hardcoded
    deepseek-v4-pro judge model (non-Kimi, satisfies family exclusion
    against the kimi-k2p6 agent; see SCOUT.md Phase 1 #2). Requires
    FIREWORKS_API_KEY — gated, not imported at module load time."""
    bossyk_src = str(BOSSYK_SRC)
    if bossyk_src not in sys.path:
        sys.path.insert(0, bossyk_src)
    from judge import PolicyAwareJudge  # type: ignore[import-not-found]
    from policy import load_policy  # type: ignore[import-not-found]

    policy = load_policy(policy_path)
    return PolicyInstrument(judge=PolicyAwareJudge(policy=policy))
