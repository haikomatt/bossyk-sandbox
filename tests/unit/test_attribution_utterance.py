"""RED-phase tests for the suppressed-utterance compliance control
(bossyk-sandbox slice 2, P7; boundary 5, prohibited_financial_promotion).

FLAG (per the RED gate): an utterance has no `ProposedAction` -- P6's
`agent_node` scores free text against `UtteranceInstrument.score(text,
history) -> Decision`; there is no tool call at all. `controls_for_step`
cannot be reused as-is (it requires a `ProposedAction`). Proposed here: a
SEPARATE, narrower entry point, `controls_for_utterance(decision: Decision)
-> list[ControlTag]`, that needs only the Decision an utterance rule
produced -- mirrors the shape of `controls_for_step`'s verdict-derived tier
(only fires on BLOCK; the block IS the compliance-relevant event) without
requiring a tool name/arguments that don't exist for a speech act.

Whether/how a suppressed utterance's ControlTag gets threaded into an
actual attested Step/trace (there is no Step for an utterance turn today --
P6 only replaces the AIMessage, see runtime/langgraph_agent.py's agent_node)
is explicitly OUT of scope this phase, per the coordinator's P6 scope note
("P6 just needs the suppression mechanism + a recorded Decision") -- these
tests exercise `controls_for_utterance` directly against synthetic
Decisions, the way `controls_for_step` is unit-tested against synthetic
ProposedActions in test_attribution.py.
"""

from __future__ import annotations

from bossyk_sandbox.compliance.attribution import controls_for_utterance
from bossyk_sandbox.compliance.frameworks import load_frameworks
from bossyk_sandbox.instruments.base import Decision, Verdict


def test_a_blocked_utterance_discharges_the_fca_conc_control() -> None:
    decision = Decision(Verdict.BLOCK, "utterance contains regulated phrase '0% finance'")

    tags = controls_for_utterance(decision)

    refs = {tag.ref for tag in tags}
    assert "fca:conc-3" in refs


def test_a_blocked_utterance_control_is_deterministic() -> None:
    # The term-list match is reproducible offline from the recorded text --
    # exactly the P7 payoff: this control is NOT merely a directional
    # mapping.
    decision = Decision(Verdict.BLOCK, "utterance contains regulated phrase '0% finance'")

    tags = controls_for_utterance(decision)

    conc_tag = next(tag for tag in tags if tag.ref == "fca:conc-3")
    assert conc_tag.discharge == "deterministic"


def test_an_allowed_utterance_discharges_no_controls() -> None:
    decision = Decision(Verdict.ALLOW, "no regulated phrase found")

    tags = controls_for_utterance(decision)

    assert tags == []


def test_utterance_control_refs_resolve_against_the_committed_catalogue() -> None:
    registry = load_frameworks()
    resolvable = {f"{f.id}:{c.id}" for f in registry.entries for c in f.controls}

    decision = Decision(Verdict.BLOCK, "utterance contains regulated phrase '0% finance'")
    tags = controls_for_utterance(decision)

    assert tags, "expected at least one control tag on a blocked utterance"
    unresolved = {tag.ref for tag in tags} - resolvable
    assert unresolved == set(), f"utterance tagger emits refs not in the catalogue: {unresolved}"
