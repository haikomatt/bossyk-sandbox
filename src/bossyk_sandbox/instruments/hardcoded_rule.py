from __future__ import annotations

from dataclasses import dataclass

from bossyk_sandbox.instruments.base import Decision, ProposedAction, Verdict

# Defaults target tau2's airline domain (see SCOUT.md). Externalised as
# constructor fields, not module constants, so a future config file/env var
# can retarget the rule without editing this class.
DEFAULT_GATED_TOOL = "cancel_reservation"
DEFAULT_REQUIRED_LOOKUP_TOOL = "get_reservation_details"
DEFAULT_KEY_ARG = "reservation_id"


@dataclass
class RequireLookupBeforeCancel:
    """The single Phase 0 policy rule.

    Blocks `gated_tool` unless an earlier step in the same session called
    `required_lookup_tool` with the same `key_arg` value.
    """

    gated_tool: str = DEFAULT_GATED_TOOL
    required_lookup_tool: str = DEFAULT_REQUIRED_LOOKUP_TOOL
    key_arg: str = DEFAULT_KEY_ARG

    def score(self, proposed: ProposedAction, history: list[ProposedAction]) -> Decision:
        if proposed.tool_name != self.gated_tool:
            return Decision(Verdict.ALLOW, f"{proposed.tool_name} is not gated by this rule")

        key_value = proposed.arguments.get(self.key_arg)
        looked_up = any(
            call.tool_name == self.required_lookup_tool
            and call.arguments.get(self.key_arg) == key_value
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
