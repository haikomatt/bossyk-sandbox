"""The discharge invariant: a `deterministic` tag must have a wired gate.

`ControlTag.discharge == "deterministic"` is a positive assertion that the
control is discharged by a REPRODUCIBLE CHECK, and it licenses the evidence
pack to drop the "directional mapping, not legal advice" disclaimer
(`compliance/attribution.py`, `_DETERMINISTIC_ACTION_TOOLS` docstring). That
claim is only true if a rule is actually wired to gate the tool in the runtime
that produces the trace.

Nothing enforced that link. `test_attribution_outreach.py` asserts the five
Tier-C tools discharge `deterministic`, but it calls `controls_for_step`
directly with a synthetic `ProposedAction` and no gate, rules or history -- it
proves the lookup table matches itself and cannot detect a missing gate. This
file supplies the missing cross-module invariant (attribution x domains x
runner), which is why it is a new file rather than an addition to any of the
three existing `test_attribution*` modules: it is not specific to a domain.

See the vault: `deterministic-discharge-defect-2026-09-04` and the distilled
claim `a-test-that-only-exercises-the-labeller-cannot-validate-the-label`.
"""

from bossyk_sandbox.compliance.attribution import _DETERMINISTIC_ACTION_TOOLS
from bossyk_sandbox.domains import _DOMAIN_BUILDERS, domain_config


def _all_gated_tools() -> set[str]:
    """Every tool gated by a fast rule in any registered domain."""
    gated: set[str] = set()
    for name in _DOMAIN_BUILDERS:
        for rule in domain_config(name).fast_rules_factory():
            gated_tool = getattr(rule, "gated_tool", None)
            if isinstance(gated_tool, str) and gated_tool:
                gated.add(gated_tool)
    return gated


def test_every_deterministic_action_tool_has_a_wired_gate() -> None:
    """A tool may only claim a deterministic discharge if some registered
    domain actually gates it. Without this, the evidence pack asserts a
    reproducible check that never ran."""
    gated = _all_gated_tools()
    unbacked = sorted(_DETERMINISTIC_ACTION_TOOLS - gated)
    assert not unbacked, (
        f"tools claim discharge='deterministic' with no wired gate: {unbacked}. "
        f"Gated across all registered domains: {sorted(gated)}. "
        "Either remove them from _DETERMINISTIC_ACTION_TOOLS (narrow the claim) "
        "or wire a rule that gates them (earn the claim)."
    )
