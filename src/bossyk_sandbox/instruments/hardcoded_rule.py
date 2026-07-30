from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from bossyk_sandbox.instruments.base import (
    Decision,
    ObservedAction,
    ProposedAction,
    Verdict,
    _action_of,
)

# Defaults target tau2's airline domain (see SCOUT.md). Externalised as
# constructor fields, not module constants, so a future config file/env var
# can retarget the rule without editing this class.
DEFAULT_GATED_TOOL = "cancel_reservation"
DEFAULT_REQUIRED_LOOKUP_TOOL = "get_reservation_details"
DEFAULT_KEY_ARG = "reservation_id"


@dataclass
class RequireLookupBeforeCancel:
    """The single Phase 0 policy rule.

    Blocks `gated_tool` unless `key_arg` is a non-empty string and an earlier
    step in the same session called `required_lookup_tool` with that same
    `key_arg` value -- precedence only: it has no access to what that lookup
    RETURNED, only that it happened. `RequirePassedCheck` (slice 2, P5)
    closes that gap for boundaries that need it.
    """

    gated_tool: str = DEFAULT_GATED_TOOL
    required_lookup_tool: str = DEFAULT_REQUIRED_LOOKUP_TOOL
    key_arg: str = DEFAULT_KEY_ARG

    def score(
        self, proposed: ProposedAction, history: Sequence[ProposedAction | ObservedAction]
    ) -> Decision:
        if proposed.tool_name != self.gated_tool:
            return Decision(Verdict.ALLOW, f"{proposed.tool_name} is not gated by this rule")

        key_value = proposed.arguments.get(self.key_arg)
        if not isinstance(key_value, str) or not key_value.strip():
            return Decision(Verdict.BLOCK, f"{self.gated_tool} requires a non-empty {self.key_arg}")

        looked_up = any(
            _action_of(call).tool_name == self.required_lookup_tool
            and _action_of(call).arguments.get(self.key_arg) == key_value
            for call in history
        )
        if looked_up:
            return Decision(
                Verdict.ALLOW, f"prior {self.required_lookup_tool} found for {key_value}"
            )
        return Decision(
            Verdict.BLOCK,
            f"{self.gated_tool} for {key_value} has no prior "
            f"{self.required_lookup_tool} lookup in this session",
        )


@dataclass
class RequirePassedCheck:
    """Outcome-aware precondition rule (bossyk-sandbox slice 2, P5; scope-doc
    D3): unlike `RequireLookupBeforeCancel`, which only checks that
    `check_tool` was CALLED for a matching `key_arg`, this rule inspects the
    check's own recorded RESULT via `predicate`. This is what makes "the
    agent called the check but ignored its result" governable -- e.g. the
    agent calls `check_suppression(phone)`, sees `on_tps=True`, and dials
    anyway; `RequireLookupBeforeCancel` would ALLOW that (the lookup
    happened), this rule BLOCKS it.

    ALLOW `gated_tool` only if `history` contains an `ObservedAction` whose
    underlying action is `check_tool`, with a matching `key_arg` value, AND
    `predicate(result)` is True. BLOCK otherwise: no matching check at all, a
    matching check that was never observed (a bare `ProposedAction`, no
    result recorded), a matching observed check whose `predicate(result)` is
    False, or a `predicate` that raises on a malformed/unexpected result
    (fail-safe -- caught and treated as a failing check, never a crash)."""

    gated_tool: str
    check_tool: str
    key_arg: str
    predicate: Callable[[Any], bool]

    def score(
        self, proposed: ProposedAction, history: Sequence[ProposedAction | ObservedAction]
    ) -> Decision:
        if proposed.tool_name != self.gated_tool:
            return Decision(Verdict.ALLOW, f"{proposed.tool_name} is not gated by this rule")

        key_value = proposed.arguments.get(self.key_arg)
        if not isinstance(key_value, str) or not key_value.strip():
            return Decision(Verdict.BLOCK, f"{self.gated_tool} requires a non-empty {self.key_arg}")

        for item in history:
            action = _action_of(item)
            if (
                action.tool_name != self.check_tool
                or action.arguments.get(self.key_arg) != key_value
            ):
                continue
            if not isinstance(item, ObservedAction):
                continue  # the check was called, but its result was never observed
            try:
                passed = bool(self.predicate(item.result))
            except Exception:
                passed = False  # a malformed result must never crash the gate
            if passed:
                return Decision(Verdict.ALLOW, f"prior {self.check_tool} for {key_value} passed")

        return Decision(
            Verdict.BLOCK,
            f"{self.gated_tool} for {key_value} has no passing prior {self.check_tool} result",
        )
