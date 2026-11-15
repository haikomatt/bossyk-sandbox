"""Decision source in the signed record (Phase C of the fail-closed audit scope).

Design rule 2: the signed record states the final verdict AND its source.
Design rule 4: source is structured, never parsed out of `reason`. Reason
strings stay human-readable; nothing reads provenance back out of them.

Decision D4: `Decision.source` is a REQUIRED field (no default), so a
forgotten source is a type error rather than a silently anonymous decision.

The contract pinned here:
- every fast rule's decisions name the rule as `"<RuleClass>:<configured id>"`,
  the same string for every decision that rule returns, and distinct from
  every other rule in the same gate;
- a gate-proxy rule's configured id is its policy id, so its source is
  exactly `"<RuleClass>:<policy.id>"`, and `HoldOnMatch` keeps the inner
  rule's source;
- `Gate.score`'s fall-through ALLOW has source `"gate:default"`, so it is
  distinguishable from an instrument's own ALLOW;
- `make_step` / `make_attested_step` put `decision_source` in step metadata
  (signed), and a human override adds `override_source: "human"`;
- the gate proxy's logged `policy_id` comes from `source`, not from the
  reason prefix;
- the proxy scorecard is byte-identical to the one `main` renders (pin).

No network, no spend: scripted fakes and fixture environments only.
"""

from __future__ import annotations

import ast
import dataclasses
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from auditk.attestation.signer import generate_keypair
from langchain_core.messages import AIMessage

from bossyk_sandbox.console.live import run_live_session
from bossyk_sandbox.domains import _DOMAIN_BUILDERS
from bossyk_sandbox.evidence.signed_trace import build_signed_trace, verify_signed_trace
from bossyk_sandbox.evidence.trace import build_trace, make_attested_step, make_step
from bossyk_sandbox.gate import Gate
from bossyk_sandbox.gateproxy.events import load_events
from bossyk_sandbox.gateproxy.policies import (
    HoldOnMatch,
    NetworkEgressRule,
    compile_instruments,
    load_policy_pack,
)
from bossyk_sandbox.gateproxy.scorecard import (
    ScorecardInputs,
    build_scorecard,
    render_scorecard_html,
)
from bossyk_sandbox.instruments.base import Decision, Instrument, ProposedAction, Verdict
from bossyk_sandbox.instruments.hardcoded_rule import RequireLookupBeforeCancel
from bossyk_sandbox.instruments.utterance_rule import ProhibitedPhraseRule
from bossyk_sandbox.runtime.langgraph_agent import build_retail_agent_session
from bossyk_sandbox.scenarios.loader import load_scenarios
from bossyk_sandbox.scenarios.runner import advice_fast_rules_observe_only
from tests.unit.gateproxy.test_scorecard import _write_pack, _write_trace
from tests.unit.test_fail_closed import (
    _ALL_TYPES_PACK,
    REGISTERED,
    _proxy_config,
    _RecordingEnv,
    _ScriptedLLM,
    _surface,
    _upstream_with,
    forwarded_tool_calls,
    hostile_arguments,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src" / "bossyk_sandbox"
EXAMPLE_PACK = REPO_ROOT / "examples" / "gateproxy" / "policy-pack.yaml"
SCORECARD_GOLDEN = REPO_ROOT / "tests" / "fixtures" / "gateproxy" / "scorecard-example-pack.html"

GATE_DEFAULT_SOURCE = "gate:default"
_UNGATED = ProposedAction("__no_rule_gates_this_tool__", {})


def _source(decision: Decision) -> object:
    """The decision's source, or None while the field does not exist (so a
    missing field fails an assertion, not an AttributeError)."""
    return getattr(decision, "source", None)


# =============================================================================
# 1. D4: `Decision.source` is required
# =============================================================================


def test_decision_declares_a_required_source_field() -> None:
    fields = {f.name: f for f in dataclasses.fields(Decision)}
    assert "source" in fields, "Decision has no source field"
    source = fields["source"]
    assert source.default is dataclasses.MISSING
    assert source.default_factory is dataclasses.MISSING


def test_a_decision_without_a_source_is_refused() -> None:
    # Built from an untyped tuple so mypy accepts the call both before the
    # field exists (Red) and after it is required (Green); the runtime
    # refusal is what this test pins.
    without_source: tuple[Any, ...] = (Verdict.ALLOW, "no source given")
    with pytest.raises(TypeError):
        Decision(*without_source)


# =============================================================================
# 2. Every fast rule names itself as the source
# =============================================================================


def _decisions_of(rule: Any) -> list[Decision]:
    """One ALLOW (a tool the rule does not gate) and, where the rule has a
    declared surface, every hostile-input decision on the tool it gates."""
    decisions = [rule.score(_UNGATED, [])]
    surface = _surface(rule)
    if surface is not None:
        tool, key, _prerequisite = surface
        for args in hostile_arguments(key).values():
            decisions.append(rule.score(ProposedAction(tool, args or {}), []))
    return decisions


def source_violations(rule: Any) -> list[str]:
    """Every decision of `rule` whose source is not `"<RuleClass>:<id>"` with
    a non-empty id, or that differs from the rule's other decisions."""
    prefix = f"{type(rule).__name__}:"
    decisions = _decisions_of(rule)
    sources = [_source(d) for d in decisions]
    violations = [
        f"{d.verdict.value} {d.reason!r}: source {s!r}"
        for d, s in zip(decisions, sources, strict=True)
        if not (isinstance(s, str) and s.startswith(prefix) and len(s) > len(prefix))
    ]
    if len(set(map(repr, sources))) > 1:
        violations.append(f"inconsistent sources {sorted(set(map(repr, sources)))}")
    return violations


@pytest.mark.parametrize(("rule_id", "rule"), REGISTERED, ids=[i for i, _ in REGISTERED])
def test_every_registered_fast_rule_names_itself_as_the_source(
    rule_id: str, rule: Instrument
) -> None:
    assert source_violations(rule) == []


def test_observe_only_advice_rules_name_themselves_too() -> None:
    # Exempt from the fail-closed contract (S4), not from provenance.
    for scenario in [None, *_advice_scenarios()]:
        for rule in advice_fast_rules_observe_only(scenario):
            assert source_violations(rule) == []


def _advice_scenarios() -> list[Any]:
    return list(load_scenarios(_DOMAIN_BUILDERS["advice"]().scenarios_path))


def distinct_source_violations(rule_sets: dict[str, list[Any]]) -> list[str]:
    """Every gate (labelled rule set) in which two rules share a source, or a
    rule has no string source. The source must identify WHICH rule decided,
    not just its class: two RequireLookupBeforeCancel rules (or two
    minimisation bands) in one gate must not share a source."""
    violations = []
    for label, rules in rule_sets.items():
        sources = [_source(rule.score(_UNGATED, [])) for rule in rules]
        if not all(isinstance(s, str) for s in sources) or len(set(sources)) != len(sources):
            violations.append(f"{label}: {sources}")
    return violations


def _gate_rule_sets(domain: str) -> dict[str, list[Any]]:
    """The fast rules one gate runs, for no scenario and each shipped one."""
    cfg = _DOMAIN_BUILDERS[domain]()
    return {
        f"{domain}:{scenario.scenario_id if scenario else 'no-scenario'}": list(
            cfg.fast_rules_factory(scenario)
        )
        for scenario in [None, *load_scenarios(cfg.scenarios_path)]
    }


@pytest.mark.parametrize("domain", sorted(_DOMAIN_BUILDERS))
def test_rules_in_one_gate_have_distinct_sources(domain: str) -> None:
    assert distinct_source_violations(_gate_rule_sets(domain)) == []


class _FixedSource(RequireLookupBeforeCancel):
    """A broken variant of a real rule whose decisions carry chosen sources:
    one for its ALLOW of an ungated tool, one for its BLOCKs of the tool it
    gates. Returns a namespace, not a `Decision`, so the guards can be
    exercised before the field exists."""

    def __init__(self, allow_source: str, block_source: str | None = None) -> None:
        self._sources = {Verdict.ALLOW: allow_source, Verdict.BLOCK: block_source or allow_source}

    def score(self, proposed: ProposedAction, history: Any) -> Any:
        verdict = Verdict.ALLOW if proposed == _UNGATED else Verdict.BLOCK
        return SimpleNamespace(verdict=verdict, reason="r", source=self._sources[verdict])


def test_distinct_source_guard_catches_two_rules_sharing_a_source() -> None:
    shared = {"gate": [_FixedSource("Rule:a"), _FixedSource("Rule:a")]}
    assert distinct_source_violations(shared) == ["gate: ['Rule:a', 'Rule:a']"]
    assert (
        distinct_source_violations({"gate": [_FixedSource("Rule:a"), _FixedSource("Rule:b")]}) == []
    )


def test_source_guard_catches_a_rule_whose_sources_disagree() -> None:
    assert source_violations(_FixedSource("_FixedSource:a", "_FixedSource:b")) != []
    assert source_violations(_FixedSource("_FixedSource:a")) == []
    assert source_violations(_FixedSource("OtherClass:a")) != []


def test_the_utterance_rule_names_itself_as_the_source() -> None:
    rule = ProhibitedPhraseRule()
    sources = {_source(rule.score(text, [])) for text in ("hello", "it is interest free")}
    assert len(sources) == 1
    (source,) = sources
    assert isinstance(source, str), f"utterance decision source {source!r}"
    assert source.startswith("ProhibitedPhraseRule:")
    assert len(source) > len("ProhibitedPhraseRule:")


# =============================================================================
# 3. Gate.score: fall-through is `gate:default`; a deciding rule is named
# =============================================================================


def test_the_gate_fall_through_allow_has_source_gate_default() -> None:
    assert _source(Gate(instruments=[]).score(_UNGATED)) == GATE_DEFAULT_SOURCE
    allowing = Gate(instruments=[RequireLookupBeforeCancel()])
    assert _source(allowing.score(_UNGATED)) == GATE_DEFAULT_SOURCE


def test_the_gate_returns_the_blocking_rules_own_source() -> None:
    rule = RequireLookupBeforeCancel()
    cancel = ProposedAction(rule.gated_tool, {rule.key_arg: "R1"})
    decision = Gate(instruments=[rule]).score(cancel)
    assert decision.verdict is Verdict.BLOCK
    assert isinstance(_source(decision), str), "the gate's decision has no source"
    assert _source(decision) == _source(rule.score(cancel, []))
    assert _source(decision) != GATE_DEFAULT_SOURCE


# =============================================================================
# 4. Gate-proxy rules: source is exactly "<RuleClass>:<policy.id>"
# =============================================================================

# One argument set per policy type that the rule MATCHES (BLOCKs).
_MATCHING_CALL: dict[str, ProposedAction] = {
    "network-egress": ProposedAction("bash", {"command": "curl https://x.example"}),
    "package-install": ProposedAction("bash", {"command": "pip install requests"}),
    "write-outside-workspace": ProposedAction("write", {"path": "/etc/passwd"}),
    "credential-read": ProposedAction("read", {"path": ".env"}),
    "protected-path-edit": ProposedAction("edit", {"path": "tests/test_provided.py"}),
}


def _all_type_rules(tmp_path: Path) -> list[Any]:
    pack_path = tmp_path / "all-types.yaml"
    pack_path.write_text(_ALL_TYPES_PACK)
    workspace = tmp_path / "ws"
    workspace.mkdir(exist_ok=True)
    return compile_instruments(load_policy_pack(pack_path), workspace_root=workspace)


def proxy_source_violations(rules: Sequence[Any]) -> list[str]:
    violations = []
    for rule in rules:
        expected = f"{type(rule).__name__}:{rule.policy.id}"
        tool = (rule.policy.config.get("tools") or ["?"])[0]
        cases = {
            "not applicable": _UNGATED,
            "match": _MATCHING_CALL[rule.policy.type],
            "non-string": ProposedAction(tool, {"command": ["curl"], "path": ["/etc"]}),
        }
        for label, proposed in cases.items():
            for wrapper, scored in (("", rule), ("HoldOnMatch ", HoldOnMatch(rule))):
                source = _source(scored.score(proposed, []))
                if source != expected:
                    violations.append(f"{wrapper}{rule.policy.id} {label}: {source!r}")
    return violations


def test_every_proxy_rule_names_its_policy_as_the_source(tmp_path: Path) -> None:
    rules = _all_type_rules(tmp_path)
    assert {r.policy.type for r in rules} == set(_MATCHING_CALL)
    assert proxy_source_violations(rules) == []


# =============================================================================
# 5. The proxy's policy_id comes from source, not the reason prefix
# =============================================================================


def _logged_events(config: Any) -> list[dict[str, Any]]:
    return load_events(config.events_path)


def _reword_egress_reason(monkeypatch: pytest.MonkeyPatch, reason: str) -> None:
    """The egress rule decides exactly as before but with a reason that no
    longer starts with its own policy id."""
    real = NetworkEgressRule.score

    def reworded(self: NetworkEgressRule, proposed: ProposedAction, history: Any) -> Decision:
        decision = real(self, proposed, history)
        if decision.verdict is Verdict.ALLOW:
            return decision
        return dataclasses.replace(decision, reason=reason)

    monkeypatch.setattr(NetworkEgressRule, "score", reworded)


def test_the_proxy_policy_id_survives_a_reason_without_the_policy_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _reword_egress_reason(monkeypatch, "network egress is not permitted")
    config = _proxy_config(tmp_path)
    forwarded_tool_calls(config, _upstream_with(("bash", '{"command": "curl https://x"}')))
    (event,) = _logged_events(config)
    assert event["verdict"] == "block"
    assert event["policy_id"] == "no-network-egress"


def test_the_proxy_policy_id_ignores_a_reason_that_names_another_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _reword_egress_reason(monkeypatch, "package-allowlist: looks like another policy")
    config = _proxy_config(tmp_path)
    forwarded_tool_calls(config, _upstream_with(("bash", '{"command": "curl https://x"}')))
    (event,) = _logged_events(config)
    assert event["policy_id"] == "no-network-egress"


_HOLD_EGRESS_PACK = _ALL_TYPES_PACK.replace(
    "    commands: [curl, wget]\n",
    "    commands: [curl, wget]\n    on_match: hold\n",
)


def test_a_held_call_is_attributed_to_its_policy_by_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert "on_match: hold" in _HOLD_EGRESS_PACK
    _reword_egress_reason(monkeypatch, "network egress needs approval")
    config = _proxy_config(tmp_path, _HOLD_EGRESS_PACK)
    forwarded_tool_calls(config, _upstream_with(("bash", '{"command": "curl https://x"}')))
    (event,) = _logged_events(config)
    assert event["verdict"] == "hold"
    assert event["policy_id"] == "no-network-egress"


# --- structural invariant: nothing reads provenance out of `reason` ----------

_REASON_PARSING_METHODS = frozenset(
    {"startswith", "endswith", "split", "rsplit", "partition", "rpartition", "removeprefix",
     "find", "rfind", "index", "rindex"}
)  # fmt: skip


def reason_parsing_violations(sources: dict[str, str]) -> list[str]:
    """Every `<expr>.reason.<method>(...)`, `<expr>.reason[...]` and
    `... in <expr>.reason` in the given modules (name -> source text):
    the ways provenance gets parsed back out of a human-readable reason."""
    violations = []
    for name, text in sources.items():
        for node in ast.walk(ast.parse(text)):
            hit: ast.AST | None = None
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in _REASON_PARSING_METHODS
                and isinstance(node.func.value, ast.Attribute)
                and node.func.value.attr == "reason"
            ):
                hit = node
            elif (
                isinstance(node, ast.Subscript)
                and isinstance(node.value, ast.Attribute)
                and node.value.attr == "reason"
            ):
                hit = node
            elif isinstance(node, ast.Compare) and any(
                isinstance(op, ast.In | ast.NotIn) for op in node.ops
            ):
                if any(
                    isinstance(c, ast.Attribute) and c.attr == "reason" for c in node.comparators
                ):
                    hit = node
            if hit is not None:
                violations.append(f"{name}:{getattr(hit, 'lineno', '?')}")
    return violations


def _src_modules() -> dict[str, str]:
    return {str(path.relative_to(REPO_ROOT)): path.read_text() for path in SRC_ROOT.rglob("*.py")}


def test_no_source_module_parses_provenance_out_of_a_reason() -> None:
    assert reason_parsing_violations(_src_modules()) == []


def test_reason_parsing_guard_catches_each_parsing_shape() -> None:
    broken = {
        "prefix.py": "x = d.reason.startswith('p:')",
        "split.py": "x = decision.reason.split(':')[0]",
        "slice.py": "x = decision.reason[:4]",
        "contains.py": "x = 'p:' in decision.reason",
    }
    assert reason_parsing_violations(broken) == [
        "prefix.py:1",
        "split.py:1",
        "slice.py:1",
        "contains.py:1",
    ]
    assert reason_parsing_violations({"ok.py": "x = f'{d.reason}'; y = d.source == 's'"}) == []


# =============================================================================
# 6. Signed steps carry decision_source (and override_source for a human)
# =============================================================================

_CANCEL = ProposedAction("cancel_reservation", {"reservation_id": "R1"})


def _gate_decision(*, block: bool) -> Decision:
    """A real gate decision: a block by RequireLookupBeforeCancel, or the
    gate's own fall-through allow."""
    gate = Gate(instruments=[RequireLookupBeforeCancel()])
    return gate.score(_CANCEL if block else _UNGATED)


def test_make_step_records_the_decision_source() -> None:
    decision = _gate_decision(block=True)
    step = make_step("t", _CANCEL, decision)
    assert _source(decision) is not None
    assert step.metadata.get("decision_source") == _source(decision)


def test_make_step_records_gate_default_for_the_fall_through() -> None:
    step = make_step("t", _UNGATED, _gate_decision(block=False))
    assert step.metadata.get("decision_source") == GATE_DEFAULT_SOURCE


def test_an_unoverridden_attested_step_records_the_automatic_source() -> None:
    auto = _gate_decision(block=True)
    step = make_attested_step("t", _CANCEL, auto_decision=auto, final_verdict=Verdict.BLOCK)
    assert isinstance(_source(auto), str), "the automatic decision has no source"
    assert step.metadata.get("decision_source") == _source(auto)
    assert "override_source" not in step.metadata
    assert step.metadata["overridden"] is False


def test_a_human_override_records_override_source_human() -> None:
    auto = _gate_decision(block=True)
    step = make_attested_step("t", _CANCEL, auto_decision=auto, final_verdict=Verdict.ALLOW)
    assert step.metadata["overridden"] is True
    assert step.metadata.get("override_source") == "human"
    # decision_source keeps naming who made the AUTOMATIC decision.
    assert step.metadata.get("decision_source") == _source(auto)


def test_a_standing_resolution_is_not_a_human_override() -> None:
    auto = _gate_decision(block=False)
    step = make_attested_step(
        "t",
        _CANCEL,
        auto_decision=auto,
        final_verdict=Verdict.BLOCK,
        mode="escalate",
        resolution_source="standing",
    )
    assert step.metadata.get("decision_source") == GATE_DEFAULT_SOURCE
    assert step.metadata["resolution_source"] == "standing"
    assert "override_source" not in step.metadata


def test_decision_source_is_covered_by_the_trace_signature(tmp_path: Path) -> None:
    step = make_step("t", _CANCEL, _gate_decision(block=True))
    assert "decision_source" in step.metadata
    priv, pub = generate_keypair(tmp_path / "k")
    signed = build_signed_trace(build_trace("t", "cfg", [step]), signer_key_path=priv)
    assert verify_signed_trace(signed, pub.read_text()) is True

    signed.trace.steps[0].metadata["decision_source"] = "forged:source"
    assert verify_signed_trace(signed, pub.read_text()) is False


# --- the live LangGraph agent threads the gate's source into its steps -------


def _retail_session_steps(calls: list[dict[str, Any]]) -> list[Any]:
    llm = _ScriptedLLM(
        [AIMessage(content="", tool_calls=[call]) for call in calls] + [AIMessage(content="done")]
    )
    session = build_retail_agent_session(llm=llm, environment=_RecordingEnv())
    return run_live_session(session).steps


def test_the_live_agent_signs_the_gates_source_on_every_step() -> None:
    session = build_retail_agent_session(llm=_ScriptedLLM([]), environment=_RecordingEnv())
    assert session.gate is not None
    cancel_rule = next(
        r for r in session.gate.instruments if isinstance(r, RequireLookupBeforeCancel)
    )
    lookup: dict[str, Any] = {"name": "get_order_details", "args": {"order_id": "#W1"}, "id": "c1"}
    cancel: dict[str, Any] = {
        "name": cancel_rule.gated_tool,
        "args": {cancel_rule.key_arg: "#W2", "reason": "no longer needed"},
        "id": "c2",
    }
    steps = _retail_session_steps([lookup, cancel])

    assert [s.metadata["gate_verdict"] for s in steps] == ["allow", "block"]
    expected_block_source = _source(
        cancel_rule.score(ProposedAction(cancel_rule.gated_tool, cancel["args"]), [])
    )
    assert [s.metadata.get("decision_source") for s in steps] == [
        GATE_DEFAULT_SOURCE,
        expected_block_source,
    ]
    assert expected_block_source is not None


# =============================================================================
# 7. Pin: the proxy scorecard renders byte-identically to `main`
# =============================================================================

# Every policy in the shipped example pack, a benign call, a non-string
# governed argument (F5) and unparseable arguments, in one response.
_SCORECARD_CALLS: tuple[tuple[str, str], ...] = (
    ("bash", '{"command": "ls"}'),
    ("bash", '{"command": "curl https://x.example"}'),
    ("bash", '{"command": "pip install requests"}'),
    ("write", '{"path": "/etc/passwd", "content": "x"}'),
    ("read", '{"path": ".env"}'),
    ("edit", '{"path": "tests/test_provided.py", "content": "x"}'),
    ("bash", '{"command": ["curl", "https://x.example"]}'),
    ("bash", "{not json"),
)

_SCORECARD_AT = datetime(2026, 11, 12, 9, 0, 0, tzinfo=UTC)


def example_pack_scorecard_html(tmp_path: Path) -> str:
    """Drive the proxy over the shipped example pack and render the scorecard."""
    config = dataclasses.replace(
        _proxy_config(tmp_path, EXAMPLE_PACK.read_text()), run_label="scorecard-pin"
    )
    forwarded_tool_calls(config, _upstream_with(*_SCORECARD_CALLS))
    priv, pub = generate_keypair(tmp_path / "pack-key")
    trace_path, pack_path = tmp_path / "trace.json", tmp_path / "evidence-pack.json"
    _write_trace(trace_path)
    _write_pack(pack_path, priv)
    card = build_scorecard(
        ScorecardInputs(
            run_label="scorecard-pin",
            task_name="example pack pin",
            trace_path=trace_path,
            evidence_pack_path=pack_path,
            gate_events_path=config.events_path,
            report_md_path=None,
            pack_public_key_pem=pub.read_text(),
            gate_public_key_pem=(tmp_path / "gate-key.ed25519.pub").read_text(),
            generated_at=_SCORECARD_AT,
        )
    )
    return render_scorecard_html(card)


def test_the_proxy_scorecard_is_byte_identical_on_the_example_pack(tmp_path: Path) -> None:
    assert example_pack_scorecard_html(tmp_path) == SCORECARD_GOLDEN.read_text()


def test_scorecard_pin_catches_a_reworded_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _reword_egress_reason(monkeypatch, "no-network-egress: egress refused")
    assert example_pack_scorecard_html(tmp_path) != SCORECARD_GOLDEN.read_text()
